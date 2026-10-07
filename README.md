# CUDA vLLM W8A8 Block FP8 Tuning Tool

[English](README.md) | [中文](README_zh.md)

A **CUDA-only vLLM W8A8 Block FP8 Triton kernel tuning tool**. The maintained entry is `benchmark_w8a8_block_fp8.py`. It tunes ordinary two-dimensional linear GEMMs, checks numerical correctness of finalists, merges all requested M values, and writes vLLM-compatible configs safely. It does not promise a speedup or tune an entire model.

## Scope and evidence

| Capability | Current scope / validation |
| --- | --- |
| Device | NVIDIA CUDA, native FP8 compute capability >= 8.9; CUDA PyTorch, vLLM and Triton required |
| Kernel | Installed vLLM `_w8a8_triton_block_scaled_mm`; FP8 E4M3FN A/B, FP32 scales; FP16, BF16 or FP32 output |
| Automatic dense adapter | `Qwen3ForCausalLM`: GQA QKV, attention output, fused gate/up, down projection |
| Partial MoE adapter | `Qwen3MoeForCausalLM`: attention plus regular dense/shared MLPs that actually exist according to the layer schedule |
| Excluded | Routed experts / fused MoE, LM head, router and shared-expert gate, Next/VL/Omni/nested configs, DeepSeek auto-detection, unknown architectures |
| Evidence | CPU shape/persistence/entry-point and source-contract tests pass. GPU compilation, numerical/performance and serving verification remain **unverified** on this machine |

Auto shape detection is architecture-aware and only enabled for explicitly supported architectures. Unsupported architectures fail instead of falling back to unrelated shapes. Names containing “Qwen” are not used to select an adapter. MoE model support does not mean routed expert kernel tuning.

Shape adapters establish the dimensions of regular linears; they do not prove that every layer is quantized or that a serving engine selects this Triton backend. If checkpoint quantization metadata exists, auto-detection accepts only dynamic `fp8` with matching `weight_block_size`. Both `ignored_layers` and `modules_to_not_convert` are checked together. Explicit non-target exclusions (LM head, embeddings, layernorms, Q/K norms, MoE router `mlp.gate`, shared-expert gate, routed-expert subtree and bias parameters) are allowed. Target projections in checkpoint or fused runtime names, broad parent scopes and unrecognized patterns fail closed with the offending exclusion and `--shape N K` guidance. The small classifier recognizes literal/scoped `*` components; it does not implement arbitrary regex patterns. Without metadata, the preview/report states that FP8 and backend selection still need runtime confirmation. Other quantization formats can use explicitly observed shapes only when they really reach the same kernel/layout.

Repository still contains historical/experimental INT8 and AWQ scripts, but they are **not part of the currently maintained and validated workflow**. They are not recommended entry points; their compatibility, correctness and runtime consumption are not maintained. No ROCm/XPU support is provided.

## Quick start

Use a CUDA environment compatible with your installed vLLM. CPU planning/tests use Python 3.10+; GPU Python, PyTorch and Triton requirements follow that vLLM build. No kernel implementation is bundled as an import fallback.

```bash
# Inspect CLI without CUDA/PyTorch/vLLM
python3 benchmark_w8a8_block_fp8.py --help

# Explicit observed per-rank (N,K): CPU-only preview
python3 benchmark_w8a8_block_fp8.py --shape 768 2048 --batch-size 17 --preview

# Architecture preview needs vLLM model-config loading, but does not initialize CUDA
python3 benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-8B --tp-size 4 --preview

# CUDA/import/symbol/signature/device preflight; failure is nonzero
bash scripts/environment_check.sh

# Small actual tuning, including default comparison, finalist remeasurement and correctness
python3 benchmark_w8a8_block_fp8.py --shape 128 256 --batch-size 17 --save-path ./tuned_configs

# Ordinary linear shapes in a Qwen3 model; confirm actual FP8 backend/checkpoint first
bash scripts/tune_qwen3.sh Qwen/Qwen3-8B 4 128 128
```

The synthetic microbenchmark directly generates quantized FP8 A/B and scales. It excludes activation quantization, model loading, attention and serving overhead. The example with an unquantized model identifier plans its linear dimensions; it does not convert the checkpoint to FP8.

## Arguments and semantics

| Argument | Meaning / default |
| --- | --- |
| `--model` | Model identifier/local config supported by the explicit adapters; exclusive with `--shape` |
| `--shape N K` | Repeatable observed **per-rank** regular-linear weight shape, already TP-sharded; no further division |
| `--tp-size`, `-tp` | Target vLLM tensor-parallel world size, default `1`; not tuning GPU count |
| `--batch-size` | Single GEMM M (flattened token rows); omitted: `1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096` |
| `--block-n`, `--block-k` | Checkpoint/runtime quantization layout, default `128,128`; powers of two >=32, not arbitrary tuning tiles |
| `--out-dtype` | `float16` (default), `half` alias, `bfloat16`, `float32`; must match target runtime |
| `--input-type` | Only `fp8` |
| `--save-path` | Default `./tuned_configs` relative to current directory; single-model wrappers default to repository `tuned_configs/` |
| `--overwrite` | Explicitly replace overlapping M values; other existing M values remain |
| `--trust-remote-code` | Default off; explicitly authorizes executing code from the model repository |
| `--preview` | Print shapes, layer sources, M, TP, layout and output dtype without CUDA |
| `--check-environment` | Check CUDA dependencies/devices and the actual required FP8 symbols/helper/signature |
| `--verify-installed` | Check saved vs installed loader results and run its public Triton wrapper; does not verify serving backend selection |
| `--seed` | Nonnegative synthetic-input seed, default `0` |
| `--measurements` | CUDA event measurement rounds, default `5` |
| `--calls-per-event` | Actual kernel calls per measured event, default `1`; >1 explicitly selects repeated-call averages |

M is the GEMM M dimension, not necessarily the number of concurrent serving requests. TP changes model shape partitions; tuning uses up to `min(visible CUDA GPUs, requested M count)` workers independently of TP. One M uses one GPU because there is no parallel M work. M values are assigned by deterministic LPT/greedy balancing using M as the cost proxy; each worker logs its M values in ascending order. Participating GPUs must be the same model. Duplicate shapes are tuned once. Empty, missing, duplicate or unexpected worker results abort before saving.

QKV uses `local_q = Q_heads / TP`, `local_kv = max(1, KV_heads / TP)`, `N = (local_q + 2*local_kv)*head_dim`, `K = hidden_size`. Attention output has `K = local_q*head_dim`, which need not equal `hidden_size/TP`. Exact Q/KV partition/replication relations are checked. A regular fused gate/up uses `N = 2*intermediate_size/TP`; it is generated only when the architecture actually constructs that MLP. Routed-expert dimensions are never inferred from `moe_intermediate_size`. Runtime K grouping and fused partition block alignment are also checked. Automatic adapters currently require square FP8 blocks: the source-audited `Fp8Config` activation grouping uses block N while the target GEMM expects block K. Non-square layouts require explicitly observed `--shape` inputs and confirmation that the real caller supplies the correct scales.

Wrappers accept `MODEL TP BLOCK_N BLOCK_K` followed by extra CLI flags. `PYTHON`, `SAVE_PATH`, `OUT_DTYPE`, `INPUT_TYPE`, `BATCH_SIZE` and explicit `TRUST_REMOTE_CODE=1` are supported environment overrides. All wrappers leave remote code off by default. To preview through a wrapper:

```bash
bash scripts/tune_custom.sh Qwen/Qwen3-8B 4 128 128 --preview
```

The batch example `examples/tune_qwen3_models.sh` isolates each model/TP task under `tuned_configs/batch/<model with slashes replaced by underscores>/tp_<TP>/`. In this batch runner, `SAVE_PATH` or `--save-path` sets the **root**; the CLI option takes precedence. For example, Qwen3-8B TP=4 writes to `tuned_configs/batch/Qwen_Qwen3-8B/tp_4/`. Repeating a task still requires explicit `--overwrite`. Install from the specific task directory you intend to use, rather than the batch root.

## Measurement and saving

Each candidate compiles/warms up with five launches. Multiple eager CUDA event rounds use a preallocated output buffer. Default measurement is one GEMM call per event, converted to microseconds without an extra divisor. Explicit `--calls-per-event >1` measures a repeated-call average: `elapsed_ms * 1000 / calls_per_event`. Repeated calls change the workload/cache behavior and can change configuration rankings; they are not interchangeable with single-call latency. Search ranks medians. The top three successful candidates and the source-audited default (when runnable) are independently remeasured on the same tensors and compared with FP32 dequantized matmul using those exact A/B/scales (`rtol=0.02`, `atol=0.02`, finite outputs required). The best remeasured median wins, so the default can win too. A default that fails with `OutOfResources` is not retried; its report entry is `baseline.status=unavailable` with the reason. A runnable default has `baseline.status=validated` and its independent measurement. Only Triton `OutOfResources` candidates are skipped; unknown compilation/runtime/numerical errors fail the task. This is a sanity check, not proof of model accuracy or globally optimal tuning.

Successful runs save a separate `reports/<UTC timestamp>-<id>.json` with seed, software/kernel source hash, device, shape/layer sources, output dtype, layout, finalist samples/errors, default comparison, and resource-failure counts. Official configs contain only M-to-launch mappings.

The filename uses the **installed vLLM device-name helper**, exactly:

```text
N={N},K={K},device_name={normalized device},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json
```

Each config save holds a filesystem lock across reading/merging/writing; JSON is written to a sibling temp file, flushed/fsynced/closed, then atomically replaced. Default: merge disjoint M; reject overlapping M before tuning. `--overwrite` replaces only requested M. Malformed or incompatible existing JSON is rejected even with `--overwrite`. Never silently reduce an existing full file to one M. `.lock` files are intentional; only copy the top-level `.json` configs. Atomicity is per file, not a transaction over all shapes; a later write failure may leave earlier complete files, with no success message. Filesystems must support advisory locking and atomic replacement.

The official filename does not distinguish output dtype, TP, model or vLLM version. Use separate `--save-path` directories for different software/kernel identities or output dtypes; merge only runs targeting the same runtime/layout. Matching filenames alone cannot prove performance compatibility with a different vLLM kernel.

## Install and verify on the target CUDA host

Use the exact output directory from your command. Single-model wrappers default to the repository `tuned_configs/` directory. For Python's default, run these commands from the repository root:

```bash
CONFIG_DIR=$(python3 -c 'from pathlib import Path; from vllm.model_executor.layers.quantization.utils import fp8_utils; print(Path(fp8_utils.__file__).resolve().parent / "configs")')
cp ./tuned_configs/*.json "$CONFIG_DIR/"

# New process: real installed loader, requested M coverage and public Triton wrapper
python3 benchmark_w8a8_block_fp8.py --shape 128 256 --batch-size 17 \
  --save-path ./tuned_configs --verify-installed
```

Configs apply only when N, K, normalized device, `fp8_w8a8` and block shape match the actual loader, and serving chooses the regular-linear Triton backend. The loader chooses the closest saved M; reports record only the M actually measured. Restart serving processes after installation because the loader caches configs. Backend selection may choose another kernel (e.g. CUTLASS/DeepGEMM/FlashInfer); copying JSON alone does not prove it is used. Verify the selected backend and real model calls on your chosen vLLM build, then compare serving performance under identical conditions. This repository makes no current end-to-end serving performance claim.

Source contracts were checked against vLLM main commit [`c741bfca70cfb777e2016f827eae31f6e215fe9f`](https://github.com/vllm-project/vllm/tree/c741bfca70cfb777e2016f827eae31f6e215fe9f) on 2026-10-08. This is a source compatibility reference, **not** a GPU-tested version guarantee. [Validation record](docs/VALIDATION.md) includes pinned kernel, adapter and loader sources and the remaining hardware gates.

Real CUDA acceptance is a **merge gate**, independent of CPU CI. The PR remains Draft until the hardware gate is satisfied. Exact preflight, small-shape tuning, installed-loader/public-wrapper checks, official model preview and 1-vs-10 call comparison commands are in [CUDA acceptance steps](docs/CUDA_VALIDATION.md).

## Tests and migration

```bash
python3 -m pytest -q
# Optional real CUDA checks: compile/tune, 3 output dtypes, M=1/17/64, N tail, official loader/wrapper
python3 -m pytest -q tests/test_gpu.py
```

The optional GPU suite isolates the installed loader's config directory in a temporary location; it does not modify installed vLLM sources. Here it is **Not executed — CUDA GPU unavailable** (PyTorch/vLLM also absent). CPU unit/source-contract and Shell subprocess tests are recorded in [docs/VALIDATION.md](docs/VALIDATION.md); no GPU speedups are invented.

`benchmark_w8a8_block_fp8_qwencoder.py` is a deprecated thin wrapper requiring the same explicit source arguments. `scripts/tune_deepseek_v3.sh` now exits nonzero with a migration message: use observed `--shape N K`. The old `...qwen3_30b.py` and `...qwen3omni_talker.py` actually run INT8 W8A8 despite their filenames; they remain marked legacy and do not redirect to FP8. [README_AWQ.md](README_AWQ.md) is historical: its custom JSON has no automatic vLLM AWQ consumer in the reviewed runtime.

## Repository layout

```text
benchmark_w8a8_block_fp8.py                # Maintained CUDA tuner/CLI
fp8_tuning.py                             # CPU shape/merge/persistence helpers
benchmark_w8a8_block_fp8_qwencoder.py       # Deprecated thin FP8 wrapper
benchmark_w8a8_block_int8.py                # Legacy, unmaintained
benchmark_awq_w4a16.py                     # Legacy, unmaintained
benchmark_w8a8_block_fp8_qwen3_30b.py        # Historical INT8, misleading filename
benchmark_w8a8_block_fp8_qwen3omni_talker.py # Historical INT8, misleading filename
scripts/                                  # Shared maintained entry points/preflight; DeepSeek deprecation
examples/tune_qwen3_models.sh              # Batch runner; any failure -> nonzero
tests/                                   # CPU, Shell, source-contract and optional GPU tests
docs/VALIDATION.md                        # Repair evidence and second review
.github/workflows/tests.yml               # CPU/Shell CI
README.md / README_zh.md / README_AWQ.md
LICENSE / NOTICE
```

Licensed under the complete Apache License 2.0 in [LICENSE](LICENSE); attribution is retained in [NOTICE](NOTICE) and SPDX headers.
