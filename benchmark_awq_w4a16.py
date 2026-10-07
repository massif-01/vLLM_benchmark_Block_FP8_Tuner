# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang quantization/tuning_block_wise_kernel.py
# LEGACY / EXPERIMENTAL / CURRENTLY NOT MAINTAINED.
# Not part of the validated CUDA W8A8 Block FP8 workflow.

"""
AWQ (W4A16) Triton Kernel Tuning Script

This script performs automated tuning of the AWQ (Activation-aware Weight Quantization)
Triton kernel for W4A16 quantization (4-bit weights, 16-bit activations) in vLLM.

The script searches for optimal kernel configurations (BLOCK_SIZE_M, BLOCK_SIZE_N,
BLOCK_SIZE_K, SPLIT_K, num_warps, num_stages) for different weight shapes and batch
sizes, enabling optimal performance for AWQ-quantized models.

Features:
    - Supports group-wise quantization with group sizes: 32, 64, 128, or -1 (per-channel)
    - Multi-GPU parallel tuning for faster completion
    - Comprehensive search space exploration
    - Model-specific weight shape configuration (default: Qwen3-30B)

Usage:
    Basic usage:
        python3 benchmark_awq_w4a16.py --tp-size 1 --group-size 128 --split-k-iters 1
    
    With custom save path:
        python3 benchmark_awq_w4a16.py --tp-size 1 --group-size 128 --split-k-iters 1 \\
            --save-path ./awq_configs
    
    Single batch size tuning:
        python3 benchmark_awq_w4a16.py --tp-size 1 --group-size 128 --split-k-iters 1 \\
            --batch-size 512

Output:
    Generates JSON configuration files for each weight shape:
        N={N},K={K},G={group_size},device_name={device_name},dtype=awq_w4a16.json
    
    These configs can be copied to:
        vllm/model_executor/layers/quantization/utils/configs/

See README_AWQ.md for detailed documentation.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm

from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton
from vllm.platforms import current_platform
from vllm.triton_utils import triton

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    # Fallback to standard argparse if FlexibleArgumentParser is not available
    import argparse
    FlexibleArgumentParser = argparse.ArgumentParser

mp.set_start_method("spawn", force=True)

assert current_platform.is_cuda(), (
    "Only support tune AWQ kernel on CUDA device."
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def awq_matmul(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
) -> torch.Tensor:
    """AWQ GEMM with configurable block sizes."""
    return awq_gemm_triton(
        input, qweight, scales, qzeros, split_k_iters,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )


def get_configs_compute_bound():
    """Generate compute-bound configurations for AWQ kernel tuning."""
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [32, 64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for split_k in [1, 2, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "SPLIT_K": split_k,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return configs


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    """AWQ GEMM wrapper."""
    return awq_gemm_triton(
        input, qweight, scales, qzeros, split_k_iters,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )


def pack_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack weights to AWQ format (4-bit quantized, packed).
    
    For tuning purposes, we generate synthetically quantized weights in the correct format.
    The actual quantization logic can be complex, so we use a simplified approach that
    generates correctly formatted tensors for kernel benchmarking.
    
    Args:
        weight: FP16/BF16 weight tensor of shape [N, K]
        group_size: Group size for quantization (32, 64, 128, or -1 for per-channel)
    
    Returns:
        qweight: Packed INT4 weights [K, N // 8], dtype=int32
        scales: Scales [num_groups, N], dtype=weight.dtype
        qzeros: Zeros [num_groups, N // 8], dtype=int32
    """
    N, K = weight.shape
    assert weight.dtype in [torch.float16, torch.bfloat16]
    assert group_size in [-1, 32, 64, 128]
    if group_size == -1:
        group_size = K

    num_groups = (K + group_size - 1) // group_size
    
    # AWQ format:
    # qweight: [K, N // 8], dtype=int32 (each int32 contains 8 packed 4-bit values)
    # scales: [K // group_size, N], dtype=weight.dtype
    # qzeros: [K // group_size, N // 8], dtype=int32
    
    # Generate synthetic quantized weights and scales
    # For tuning, we just need correctly shaped tensors
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max,
        (K, N // 8),
        dtype=torch.int32,
        device=weight.device
    )
    
    scales = torch.rand(
        (num_groups, N),
        dtype=weight.dtype,
        device=weight.device
    ) * 0.1 + 0.01  # Reasonable scale range
    
    qzeros = torch.randint(
        0, torch.iinfo(torch.int32).max,
        (num_groups, N // 8),
        dtype=torch.int32,
        device=weight.device
    )
    
    return qweight, scales, qzeros


def benchmark_config(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    config: dict[str, Any],
    num_iters: int = 10,
) -> float:
    """Benchmark AWQ GEMM kernel."""
    M, K = a.shape
    split_k_iters = config.get("SPLIT_K", 1)
    block_size_m = config.get("BLOCK_SIZE_M", 32)
    block_size_n = config.get("BLOCK_SIZE_N", 32)
    block_size_k = config.get("BLOCK_SIZE_K", 32)

    torch.cuda.synchronize()
    # Warmup
    for _ in range(5):
        awq_gemm(a, qweight, scales, qzeros, split_k_iters, block_size_m, block_size_n, block_size_k)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        awq_gemm(a, qweight, scales, qzeros, split_k_iters, block_size_m, block_size_n, block_size_k)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / len(latencies) * 1000  # us
    return avg


def tune(
    M: int,
    N: int,
    K: int,
    group_size: int,
    search_space: list[dict[str, Any]],
    out_dtype: torch.dtype,
    split_k_iters: int,
):
    """Tune AWQ kernel for given weight shape."""
    # AWQ requires N to be divisible by 8 (packing factor)
    if N % 8 != 0:
        # Round up to nearest multiple of 8
        N = ((N + 7) // 8) * 8
    
    # Create random FP16/BF16 activation tensor
    a = torch.randn(M, K, dtype=out_dtype, device="cuda")

    # Create AWQ quantized weights
    weight_fp16 = torch.randn(N, K, dtype=out_dtype, device="cuda")
    qweight, scales, qzeros = pack_weight(weight_fp16, group_size)

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        if config.get("SPLIT_K", 1) != split_k_iters:
            continue
        try:
            kernel_time = benchmark_config(
                a, qweight, scales, qzeros, group_size, config, num_iters=10
            )
        except triton.runtime.autotuner.OutOfResources:
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    assert best_config is not None
    return best_config


def save_configs(
    N: int,
    K: int,
    group_size: int,
    configs: dict[int, dict[str, Any]],
    save_path: str,
) -> None:
    """Save AWQ kernel configurations."""
    os.makedirs(save_path, exist_ok=True)
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},G={group_size},device_name={device_name},dtype=awq_w4a16.json"
    )
    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")
    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_weight_shapes(tp_size):
    # NOTE: Weight shapes for Qwen3 models (default for 30B)
    # Use Qwen3-Coder-30B-A3B-Instruct configuration
    # Config: hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=128,
    #         moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8
    
    # Attention QKV projection: (num_heads * head_dim + 2 * num_kv_heads * head_dim, hidden_size)
    # = (32 * 128 + 2 * 4 * 128, 2048) = (5120, 2048)
    # Attention O projection: (hidden_size, num_heads * head_dim)
    # = (2048, 32 * 128) = (2048, 4096)
    # MoE gate_up_proj (merged): (2 * moe_intermediate_size, hidden_size)
    # = (2 * 768, 2048) = (1536, 2048)
    # MoE down_proj: (hidden_size, moe_intermediate_size)
    # = (2048, 768)
    
    # N can TP (RowParallelLinear - split output dimension)
    n_tp = [
        (1536, 2048),  # MoE gate_up_proj (merged)
    ]
    
    # K can TP (ColumnParallelLinear - split input dimension)
    k_tp = [
        (5120, 2048),  # Attention QKV projection
        (2048, 4096),  # Attention O projection
        (2048, 768),   # MoE down_proj
    ]

    weight_shapes = []
    for n_t in n_tp:
        weight_shapes.append((n_t[0] // tp_size, n_t[1]))
    for k_t in k_tp:
        weight_shapes.append((k_t[0], k_t[1] // tp_size))
    return weight_shapes


def tune_on_gpu(args_dict: dict[str, Any]):
    """Run tuning on a specific GPU."""
    gpu_id = args_dict["gpu_id"]
    batch_sizes = args_dict["batch_sizes"]
    weight_shapes = args_dict["weight_shapes"]
    args = args_dict["args"]

    torch.cuda.set_device(gpu_id)
    block_size_m = args.block_size_m
    block_size_n = args.block_size_n
    block_size_k = args.block_size_k
    split_k_iters = args.split_k_iters
    group_size = args.group_size
    out_dtype = DTYPE_MAP[args.out_dtype]
    search_space = get_configs_compute_bound()
    search_space = [c for c in search_space if group_size % c.get("BLOCK_SIZE_K", 32) == 0]

    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        best_configs = {}
        for batch_size in batch_sizes:
            best_configs[batch_size] = tune(
                batch_size, N, K, group_size, search_space, out_dtype, split_k_iters
            )
        save_configs(N, K, group_size, best_configs, args.save_path)


def distribute_batch_sizes(batch_sizes: list[int], num_gpus: int) -> list[list[int]]:
    """Distribute batch sizes across GPUs."""
    batches_per_gpu = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    """Main tuning function."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for parallel tuning")

    if args.batch_size is None:
        batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1

    weight_shapes = get_weight_shapes(args.tp_size)
    batches_per_gpu = distribute_batch_sizes(batch_sizes, num_gpus)
    process_args = []
    for gpu_id in range(num_gpus):
        process_args.append(
            {
                "gpu_id": gpu_id,
                "batch_sizes": batches_per_gpu[gpu_id],
                "weight_shapes": weight_shapes,
                "args": args,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.map(tune_on_gpu, process_args)
    print("Multi-GPU tuning completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""
Tune AWQ (W4A16) Triton kernel for Qwen3 models.
    python3 benchmark_awq_w4a16.py --tp-size 1 --group-size 128 --split-k-iters 1
Then copy configs to model_executor/layers/quantization/utils/configs
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--split-k-iters", type=int, default=1)
    parser.add_argument("--block-size-m", type=int, default=32)
    parser.add_argument("--block-size-n", type=int, default=32)
    parser.add_argument("--block-size-k", type=int, default=32)
    parser.add_argument("--out-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--save-path", type=str, default="./")
    args = parser.parse_args()

    main(args)

