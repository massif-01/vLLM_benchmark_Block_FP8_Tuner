# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang quantization/tuning_block_wise_kernel.py
# Repository modifications: strict Qwen3 shape planning, safe merging and validation.
"""CUDA-only tuner for the installed vLLM W8A8 Block FP8 Triton kernel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import uuid
from typing import Any

from fp8_tuning import (DEFAULT_BATCH_SIZES, atomic_json, config_filename,
                        distribute_batch_sizes, load_model_shapes, merge_results,
                        positive, read_configs, save_configs, timing_stats,
                        unique_shapes, validate_launch, validate_model_quantization)

# GPU dependencies are loaded only for tuning/checking. Spawn workers load their own runtime.
torch = triton = current_platform = _w8a8_triton_block_scaled_mm = None


def load_runtime():
    global torch, triton, current_platform, _w8a8_triton_block_scaled_mm
    import torch as _torch
    import vllm
    from vllm.platforms import current_platform as platform
    from vllm.triton_utils import triton as _triton
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _w8a8_triton_block_scaled_mm as kernel, get_w8a8_block_fp8_configs,
        w8a8_triton_block_scaled_mm,
    )
    from vllm.utils.platform_utils import get_device_name_as_file_name
    from vllm.transformers_utils.config import get_config
    if not platform.is_cuda() or not _torch.cuda.is_available():
        raise RuntimeError('CUDA-only tuning requires NVIDIA CUDA GPUs and CUDA PyTorch/vLLM')
    expected = ['A','B','C','As','Bs','M','N','K','group_n','group_k',
                'stride_am','stride_ak','stride_bk','stride_bn','stride_cm','stride_cn',
                'stride_As_m','stride_As_k','stride_Bs_k','stride_Bs_n',
                'BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K','GROUP_SIZE_M']
    if list(kernel.arg_names) != expected:
        raise RuntimeError('Installed vLLM FP8 kernel signature changed; this tuner is incompatible')
    get_device_name_as_file_name()
    torch, triton, current_platform, _w8a8_triton_block_scaled_mm = _torch, _triton, platform, kernel
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'vllm': vllm.__version__, 'triton': triton.__version__,
            'kernel': 'vllm.model_executor.layers.quantization.utils.fp8_utils._w8a8_triton_block_scaled_mm',
            'kernel_sha256': hashlib.sha256(kernel.src.encode()).hexdigest(),
            'kernel_file': str(Path(kernel.fn.__code__.co_filename).resolve()),
            'loader_file': str(Path(get_w8a8_block_fp8_configs.__wrapped__.__code__.co_filename).resolve())}


def check_devices(device_ids):
    for device_id in device_ids:
        capability = torch.cuda.get_device_capability(device_id)
        if capability < (8, 9):
            raise RuntimeError(f'GPU {device_id} requires compute capability >= 8.9 for native FP8; found {capability}')
        config_filename(128,128,128,128,device_id)


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype=None,
    output=None,
) -> torch.Tensor:
    """Launch the installed vLLM kernel; optionally reuse the timing output buffer."""
    if output_dtype is None:
        output_dtype = torch.float16
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = output if output is not None else A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if A.dtype == torch.float8_e4m3fn:
        kernel = _w8a8_triton_block_scaled_mm
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel. / 目前仅支持调优 w8a8 block fp8 内核")

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def get_configs_compute_bound(block_k=128):
    configs = []
    for stages in [2, 3, 4, 5]:
        for bm in [16, 32, 64, 128, 256]:
            for bk in [32, 64, 128]:
                if block_k % bk:
                    continue  # Every dot tile must stay within one K scale group.
                for bn in [32, 64, 128, 256]:
                    for warps in [4, 8]:
                        for group in [1, 16, 32, 64]:
                            configs.append(dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                                                BLOCK_SIZE_K=bk, GROUP_SIZE_M=group,
                                                num_warps=warps, num_stages=stages))
    return configs


def default_config(block_n, block_k):
    # Default from the source-audited vLLM CUDA w8a8_triton_block_scaled_mm wrapper.
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=32, num_warps=4, num_stages=2)


def benchmark_config(A, B, As, Bs, block_size, config, out_dtype, num_iters=5, calls_per_event=1):
    C = A.new_empty((A.shape[0], B.shape[0]), dtype=out_dtype)
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype, output=C)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_ms = []
    for _ in range(num_iters):
        start.record()
        for _ in range(calls_per_event):
            run()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
    return timing_stats(elapsed_ms, calls_per_event)


def reference_matmul(A, B, As, Bs, block_size, out_dtype):
    """FP32 dequantized matmul of the SAME quantized tensors; bound weight workspace."""
    bn, bk = block_size
    M, K = A.shape
    N = B.shape[0]
    torch.backends.cuda.matmul.allow_tf32 = False
    activation = A.float() * As.repeat_interleave(bk, dim=1)[:, :K]
    reference = torch.empty((M, N), dtype=out_dtype, device=A.device)
    for start in range(0, N, 256):
        stop = min(start + 256, N)
        indices = torch.arange(start, stop, device=B.device) // bn
        scales = Bs[indices].repeat_interleave(bk, dim=1)[:, :K]
        weight = B[start:stop].float() * scales
        reference[:, start:stop] = (activation @ weight.t()).to(out_dtype)
    return reference


def check_correctness(A, B, As, Bs, block_size, config, out_dtype, reference):
    actual = w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)
    # Compare implementation error, not FP8 quantization error; NaN/Inf always fail.
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError('Non-finite FP8 output/reference')
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    return {'rtol': 0.02, 'atol': 0.02, 'max_abs_error': (actual.float()-reference.float()).abs().max().item()}


def tune(M, N, K, args, search_space):
    # Same seed/data for all candidates, baseline and independent finalist remeasurements.
    torch.manual_seed(args.seed)
    A = ((torch.rand(M,K,device='cuda')-0.5)*896).to(torch.float8_e4m3fn)
    B = ((torch.rand(N,K,device='cuda')-0.5)*896).to(torch.float8_e4m3fn)
    As = (torch.rand(M,triton.cdiv(K,args.block_k),device='cuda')+0.1)*0.01
    Bs = (torch.rand(triton.cdiv(N,args.block_n),triton.cdiv(K,args.block_k),device='cuda')+0.1)*0.01
    out_dtype = getattr(torch, args.out_dtype)
    block_size = [args.block_n,args.block_k]
    baseline = default_config(*block_size)
    candidates = [baseline] + [c for c in search_space if c != baseline]
    ranked = []
    resource_failures = 0
    baseline_failure = None
    for config in candidates:
        try:
            stats = benchmark_config(A,B,As,Bs,block_size,config,out_dtype,args.measurements,args.calls_per_event)
        except triton.runtime.autotuner.OutOfResources as exc:
            resource_failures += 1
            if config == baseline:
                baseline_failure = {'type': 'OutOfResources', 'message': str(exc)}
            continue
        ranked.append((stats['median_us'], config))
    if not ranked:
        raise RuntimeError(f'No valid configuration for M={M}, N={N}, K={K}')
    finalists = [c for _,c in sorted(ranked,key=lambda x:x[0])[:3]]
    if baseline_failure is None and baseline not in finalists:
        finalists.append(baseline)
    reference = reference_matmul(A,B,As,Bs,block_size,out_dtype)
    measured = []
    for config in finalists:
        correctness = check_correctness(A,B,As,Bs,block_size,config,out_dtype,reference)
        stats = benchmark_config(A,B,As,Bs,block_size,config,out_dtype,args.measurements,args.calls_per_event)
        measured.append({'config':config,'timing':stats,'correctness':correctness})
    winner = min(measured,key=lambda r:r['timing']['median_us'])
    baseline_result = (
        {'config': baseline, 'status': 'unavailable', 'reason': baseline_failure}
        if baseline_failure is not None else
        {'config': baseline, 'status': 'validated',
         'measurement': next(r for r in measured if r['config'] == baseline)}
    )
    print(f'GPU {torch.cuda.current_device()}: M={M}, N={N}, K={K}: {winner["timing"]["median_us"]:.3f} us',flush=True)
    return winner['config'], {'M':M,'N':N,'K':K,'winner':winner, 'finalists':measured,
                              'searched':len(candidates),'resource_failures':resource_failures,
                              'baseline':baseline_result}


def tune_on_gpu(task):
    identity = load_runtime()
    gpu_id, args = task['gpu_id'], task['args']
    torch.cuda.set_device(gpu_id)
    check_devices([gpu_id])
    configs, measurements = {}, []
    search = get_configs_compute_bound(args.block_k)
    for N,K in task['weight_shapes']:
        configs[(N,K)] = {}
        for M in task['batch_sizes']:
            config, report = tune(M,N,K,args,search)
            configs[(N,K)][M] = config
            measurements.append(report)
    return {'configs':configs,'measurements':measurements,'identity':identity,'gpu_id':gpu_id}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--model', help='Only explicit Qwen3/Qwen3Moe architecture adapters')
    source.add_argument('--shape', nargs=2, type=int, action='append', metavar=('N','K'),
                        help='Observed per-TP-rank regular-linear shape; repeatable')
    parser.add_argument('--tp-size','-tp',type=int,default=1,help='Target vLLM TP world size, independent of tuning GPUs')
    parser.add_argument('--input-type',choices=['fp8'],default='fp8')
    parser.add_argument('--out-dtype',choices=['float32','float16','half','bfloat16'],default='float16')
    parser.add_argument('--block-n',type=int,default=128)
    parser.add_argument('--block-k',type=int,default=128)
    parser.add_argument('--batch-size',type=int,help='Single GEMM M / flattened token rows')
    parser.add_argument('--save-path',default='./tuned_configs')
    parser.add_argument('--trust-remote-code',action='store_true',help='Explicitly authorize executing model repository code')
    parser.add_argument('--overwrite',action='store_true',help='Replace overlapping M, preserve all other existing M')
    parser.add_argument('--preview',action='store_true',help='Print task JSON without loading CUDA')
    parser.add_argument('--check-environment',action='store_true')
    parser.add_argument('--verify-installed',action='store_true',help='Check installed vLLM loader and run its public Triton wrapper against saved config')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--measurements',type=int,default=5)
    parser.add_argument('--calls-per-event',type=int,default=1,
                        help='Kernel calls per CUDA event; >1 explicitly measures repeated-call averages')
    return parser


def plan(args):
    for name in ('tp_size','block_n','block_k','measurements','calls_per_event'):
        positive(getattr(args,name),name)
    if args.seed < 0:
        raise ValueError('seed must be nonnegative')
    for name in ('block_n','block_k'):
        value = getattr(args,name)
        if value < 32 or value & (value-1):
            raise ValueError(f'{name} must be a power of two >=32')
    if args.out_dtype == 'half':
        args.out_dtype = 'float16'
    sizes = DEFAULT_BATCH_SIZES if args.batch_size is None else [positive(args.batch_size,'batch_size')]
    if args.model:
        config, shapes, sources = load_model_shapes(args.model,args.tp_size,args.trust_remote_code)
        note = validate_model_quantization(config,args.block_n,args.block_k)
    elif args.shape:
        shapes = unique_shapes(args.shape)
        sources = [{'shape':list(s),'layer':'user-observed per-rank regular linear'} for s in shapes]
        note = 'Explicit per-rank shapes are not divided by TP; user must confirm actual FP8/backend/layout.'
    else:
        raise ValueError('Specify --model or --shape N K; no default/fallback architecture')
    validate_plan_layout(shapes,sources,args,config if args.model else None)
    return {'shapes':shapes,'M':sizes,'sources':sources,'note':note,'tp_size':args.tp_size,
            'block_shape':[args.block_n,args.block_k],'out_dtype':args.out_dtype}


def validate_plan_layout(shapes, sources, args, config=None):
    from fp8_tuning import field
    for N,K in shapes:
        if K % args.block_k:
            raise ValueError(f'K={K} must be divisible by block_k={args.block_k} for runtime activation quantization')
    if config is None:
        return
    q = field(config,'num_attention_heads')
    kv = field(config,'num_key_value_heads')
    head = field(config,'head_dim') or field(config,'hidden_size') // q
    for item in sources:
        N,_ = item['shape']
        layer = item['layer']
        if layer.endswith('qkv_proj'):
            widths = [q // args.tp_size * head, max(1,kv // args.tp_size) * head]
        elif layer.endswith('gate_up_proj'):
            widths = [N // 2]
        else:
            continue
        if any(w % args.block_n for w in widths):
            raise ValueError(f'{layer} fused partition widths {widths} must be divisible by block_n={args.block_n} exactly')


def verify_installed(args, tasks):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_w8a8_block_fp8_configs, w8a8_triton_block_scaled_mm)
    torch.cuda.set_device(0)
    check_devices([0])
    get_w8a8_block_fp8_configs.cache_clear()
    for N,K in tasks['shapes']:
        path = Path(args.save_path)/config_filename(N,K,args.block_n,args.block_k)
        expected = read_configs(path,args.block_k)
        loaded = get_w8a8_block_fp8_configs(N,K,args.block_n,args.block_k)
        if loaded != expected or not set(tasks['M']) <= set(loaded or {}):
            raise RuntimeError(f'Installed vLLM loader does not match {path}; install configs and restart')
        for M in tasks['M']:
            torch.manual_seed(args.seed)
            A = ((torch.rand(M,K,device='cuda')-.5)*896).to(torch.float8_e4m3fn)
            B = ((torch.rand(N,K,device='cuda')-.5)*896).to(torch.float8_e4m3fn)
            As = (torch.rand(M,triton.cdiv(K,args.block_k),device='cuda')+.1)*.01
            Bs = (torch.rand(triton.cdiv(N,args.block_n),triton.cdiv(K,args.block_k),device='cuda')+.1)*.01
            dtype = getattr(torch,args.out_dtype)
            reference = reference_matmul(A,B,As,Bs,[args.block_n,args.block_k],dtype)
            actual = w8a8_triton_block_scaled_mm(A,B,As,Bs,[args.block_n,args.block_k],dtype)
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise RuntimeError('Non-finite output in installed runtime wrapper')
            torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
    print('Installed vLLM loader and public Triton wrapper verified (not a model-serving backend check)')


def main(args):
    if args.check_environment:
        identity = load_runtime()
        check_devices(range(torch.cuda.device_count()))
        print(json.dumps(identity,indent=2))
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}, SM {torch.cuda.get_device_capability(i)}')
        return
    tasks = plan(args)
    if args.preview:
        print(json.dumps(tasks,indent=2))
        return
    identity = load_runtime()
    if args.verify_installed:
        verify_installed(args,tasks)
        return
    assignments = distribute_batch_sizes(tasks['M'],torch.cuda.device_count())
    check_devices(range(len(assignments)))
    names = {torch.cuda.get_device_name(i) for i in range(len(assignments))}
    if len(names) != 1:
        raise RuntimeError(f'Multi-GPU tuning requires identical GPU models: {sorted(names)}')
    torch.cuda.set_device(0)
    # Catch pre-existing conflicts before an expensive search; writer rechecks under lock.
    for N,K in tasks['shapes']:
        path = Path(args.save_path)/config_filename(N,K,args.block_n,args.block_k)
        if path.exists():
            previous = read_configs(path,args.block_k)
            if previous.keys() & set(tasks['M']) and not args.overwrite:
                raise FileExistsError(f'{path}: overlapping M; use --overwrite')
    process_args = [dict(gpu_id=i,batch_sizes=b,weight_shapes=tasks['shapes'],args=args)
                    for i,b in enumerate(assignments)]
    # One task has no parallel work; use one GPU without creating a redundant process.
    if len(process_args) == 1:
        workers = [tune_on_gpu(process_args[0])]
    else:
        with mp.get_context('spawn').Pool(len(process_args)) as pool:
            workers = pool.map(tune_on_gpu,process_args)  # Exceptions propagate; never print success after failure.
    merged = merge_results([w['configs'] for w in workers],assignments,tasks['shapes'])
    report = {'status':'kernel_correctness_validated', 'identity':identity,'device':next(iter(names)),
              'plan':tasks,'arguments':vars(args),'timing_boundary':'eager CUDA events, preallocated output, excludes quantization/reference',
              'workers':workers,'files':[]}
    # Shape-keyed dicts cannot be JSON encoded; configs are stored only in loader files.
    report['workers'] = [{k:v for k,v in w.items() if k != 'configs'} for w in workers]
    for N,K in tasks['shapes']:
        path = Path(args.save_path)/config_filename(N,K,args.block_n,args.block_k)
        save_configs(path,merged[(N,K)],args.block_k,args.overwrite)
        report['files'].append(str(path.resolve()))
    report_path = Path(args.save_path)/'reports'/ (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]+'.json')
    atomic_json(report_path,report)
    print(f'Tuning completed. Configs: {Path(args.save_path).resolve()}; validation report: {report_path}')


def cli():
    parser = build_parser()
    args = parser.parse_args()
    if sum((args.preview,args.check_environment,args.verify_installed)) > 1:
        parser.error('--preview, --check-environment and --verify-installed are mutually exclusive')
    try:
        main(args)
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())
