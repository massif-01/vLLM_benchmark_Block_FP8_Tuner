# vLLM Block FP8 Tuner

**CUDA W8A8 Block FP8 · Triton kernel tuning for regular linear layers**

[English](README.md) · [简体中文](README_zh.md)

Find launch configurations for vLLM's W8A8 block-FP8 Triton matrix-multiplication kernel, validate the selected results numerically, and export them in the JSON format read by vLLM. Tuning work can be distributed across multiple NVIDIA GPUs without workers overwriting each other's results.

This is an **independent tuning utility**, not an official vLLM component. It measures isolated GEMMs—not full-model inference—and does not guarantee a serving speedup.

> **Validation status:** CPU, CLI, and shell regression checks have passed on the reviewed PR. Real CUDA compilation, numerical tuning, installed-loader integration, and serving performance are **not yet verified**. Do not treat generated configurations as production-validated until the [CUDA acceptance procedure](docs/CUDA_VALIDATION.md) has passed on your target system.

## What it does

```text
Model config ──→ supported Qwen3 per-rank linear shapes ──┐
                                                        ├─→ tune (M, N, K)
Observed per-rank (N, K) ────────────────────────────────┘       │
                                                              ▼
                                             benchmark Triton launch candidates
                                                              │
                                                              ▼
                                            remeasure + numerical correctness
                                                              │
                                                              ▼
                                         merge results from all GPU workers
                                                              │
                                                              ▼
                                            write vLLM-compatible config JSON
```

- **Model-aware planning.** Derives actual tensor-parallel shapes for `Qwen3ForCausalLM` and `Qwen3MoeForCausalLM`, including grouped-query attention and the regular dense/shared MLPs that are present.
- **Manual shape mode.** Tunes explicitly observed per-rank `(N, K)` dimensions when automatic planning is not appropriate.
- **Multi-GPU tuning.** Distributes the requested `M` values across matching GPUs, merges results in the parent process, and checks every expected shape and `M` before saving.
- **Configuration protection.** Merges disjoint `M` entries, refuses overlapping entries by default, and uses a file lock plus atomic replacement when updating each JSON file.
- **Measured, not assumed.** Compares shortlisted kernels with an FP32 dequantized reference computed from the same quantized inputs; records timing, correctness, and software/kernel identity separately from the loader config.

### Scope

| Supported | Outside the maintained scope |
| --- | --- |
| NVIDIA CUDA GPUs with compute capability **8.9 or newer** | Earlier GPUs, ROCm, XPU, or CPU tuning |
| Installed vLLM W8A8 block-FP8 **Triton regular-linear** kernel | Routed/fused MoE expert kernel tuning |
| Explicit Qwen3 dense and Qwen3-MoE *regular-linear* shape adapters | Generic architecture guessing; Qwen3-Next/VL/Omni and DeepSeek automatic planning |
| FP8 E4M3FN inputs and FP32 scales; FP16, BF16, or FP32 outputs | INT8 and AWQ tuning (legacy implementations removed) |

**Important:** A Qwen3-MoE adapter does **not** tune routed experts. Nor does finding a linear shape prove that the running model uses block-FP8 or selects the Triton backend. Runtime backend selection must be verified separately.

## Requirements

- A Python environment with **CUDA-enabled PyTorch, Triton, and a compatible vLLM installation**. Follow the requirements of your installed vLLM release; this repository does not provide a substitute kernel or a separate dependency compatibility matrix.
- An NVIDIA GPU with **SM 8.9+** for actual tuning. For example, RTX 4090 (SM 8.9) and H100 (SM 9.0) meet this requirement; RTX A6000 (SM 8.6) and Jetson AGX Orin (SM 8.7) do not.
- The installed vLLM private FP8 kernel, configuration loader, and device-name helper must match the contracts checked by the tuner. Run preflight after every relevant vLLM upgrade.

The CPU-only `--help` and explicit-shape `--preview` commands do not require CUDA, PyTorch, or vLLM. Model-aware preview requires vLLM's model-config loader and dtype resolver, but does not load model weights or initialize a tuning GPU.

## Quick start

Run these commands from the repository root.

**1. Check the target CUDA environment**

```bash
bash scripts/environment_check.sh
```

This checks the necessary imports, kernel signature, device naming, and GPU capability. A nonzero exit means the environment is not ready for tuning.

**2. Preview one known GEMM shape**

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --preview
```

Here the weight is `N × K = 128 × 256`, while `M = 17` is the number of flattened input rows processed by the GEMM. Preview displays the task plan; it does **not** compile or benchmark a kernel.

**3. Tune that one shape**

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --save-path ./tuned_configs/quickstart
```

The tuner searches a substantial grid of launch configurations even for one shape and one `M`; start with this limited workload before attempting a full sweep. It writes a vLLM-format JSON and a separate report only after successful tuning and validation. **Generating a file does not install it into vLLM.**

### Tune shapes inferred from a Qwen3 model

Preview the official Qwen3-Coder FP8 model's regular-linear shapes at target tensor-parallel size 4:

```bash
python3 benchmark_w8a8_block_fp8.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tp-size 4 \
  --preview
```

The model's configuration is loaded, but its weights are not. `--out-dtype auto` is the default in model mode: the tool delegates dtype selection to the **installed vLLM resolver**. For this BF16 model, the expected dtype on a compatible target CUDA platform is `bfloat16`; an explicitly supplied `--out-dtype` takes precedence.

To run a single-`M` sweep through the convenience wrapper after checking the plan and backend:

```bash
BATCH_SIZE=17 bash scripts/tune_qwen3_coder.sh
```

This wrapper defaults to the Qwen3-Coder FP8 model, target TP=4, and 128×128 FP8 blocks. The `BATCH_SIZE` environment variable limits it to one `M`; without it, the tuner uses the full default `M` grid. `scripts/tune_qwen3.sh` and `scripts/tune_custom.sh` are also available for supported configurations.

**Model planning is not quantization conversion.** For example, `--model Qwen/Qwen3-8B` can describe the regular-linear dimensions of an unquantized checkpoint, but does not turn that checkpoint into FP8 or prove that your deployed model uses this kernel.

## Choose the right tuning mode

| | `--model MODEL` | `--shape N K` |
| --- | --- | --- |
| Source of `(N, K)` | Supported Qwen3 model configuration and target `--tp-size` | Dimensions already observed for one target TP rank |
| Output dtype | `auto` by default through the installed vLLM resolver | **Must** be explicit: `float16`, `bfloat16`, or `float32` |
| FP8 layout | Checks supported checkpoint metadata and the automatic adapter's block/alignment limits | You must confirm that the real runtime uses the same quantization layout and Triton kernel |
| Best for | Supported Qwen3 regular-linear configurations | Unsupported model architectures or individual known GEMMs |

In either mode, `M` is the flattened token-row count passed to a GEMM—not necessarily the serving request batch size. `--tp-size` describes how a target model's weights are sharded; **it does not set the number of GPUs doing the tuning**.

For model mode, `auto` dtype is **platform-dependent**. A preview run on a CPU-only machine may resolve a different dtype from the intended NVIDIA CUDA runtime; the final dtype must be confirmed on the target CUDA system. For manual shape mode, omitting `--out-dtype` is an error rather than an implicit FP16 choice.

Automatic shape planning is intentionally narrow. It validates Q/KV head partitioning or replication, fused projection widths, the presence of dense/shared MLPs, and applicable FP8 block alignment. Unknown architectures, broad quantization exclusions, or incompatible checkpoint layouts fail with guidance to use observed shapes rather than guessing. The automatic adapters currently require **square FP8 block sizes**; explicit shapes do not waive the requirement to verify the actual runtime layout.

## Output and reuse

A completed run produces files like:

```text
tuned_configs/quickstart/
├── N=128,K=256,device_name=GPU_NAME,dtype=fp8_w8a8,block_shape=[128,128].json
├── N=128,K=256,device_name=GPU_NAME,dtype=fp8_w8a8,block_shape=[128,128].json.lock
└── reports/
    └── <UTC-timestamp>-<run-id>.json
```

The top-level `.json` is the **vLLM loader config**: a mapping from integer-valued `M` keys to Triton launch parameters. The separate report contains the plan, measurements, correctness results, seed, device, and software/kernel identity. `.lock` files are internal to the writer; do not install them.

- By default, rerunning a file with **new, disjoint** `M` values preserves its existing entries. Repeating an existing `M` fails rather than silently replacing a tuned result.
- `--overwrite` replaces **only the requested** overlapping `M` values, retaining other entries.
- Participating GPUs must be the same model. `M` values are assigned using deterministic load balancing; worker results are checked for missing, extra, or duplicate entries before the parent writes them.
- Each file is written atomically under a filesystem lock. A run touching multiple shapes is **not** a transaction: if a later file fails, earlier completed files can remain, but the overall command exits unsuccessfully.

**Separate incompatible tuning contexts.** vLLM's filename contains `(N, K)`, device name, FP8 type, and block shape, **not** the model, TP, output dtype, or vLLM/kernel version. Use distinct `--save-path` directories for different output dtypes, kernel builds, or other incompatible contexts. The tool does not automatically reject cross-run provenance mismatches that share a filename.

### Inspect, install, and verify safely

Generated JSON is inert until placed in the installed vLLM configuration directory. **Use a disposable vLLM environment for validation**: installing a file changes which tuning config that environment's Triton wrapper may load. Do not overwrite an existing installation blindly.

The following example installs **one file only if no file with that name already exists**:

```bash
CONFIG_DIR=$(python3 -c 'from pathlib import Path; from vllm.model_executor.layers.quantization.utils import fp8_utils; print(Path(fp8_utils.__file__).resolve().parent / "configs")')
SOURCE=$(find ./tuned_configs/quickstart -maxdepth 1 -type f -name 'N=*.json' -print -quit)

if [ -z "$SOURCE" ]; then
  echo "No generated config found" >&2
else
  mkdir -p "$CONFIG_DIR"
  TARGET="$CONFIG_DIR/$(basename "$SOURCE")"
  if [ -e "$TARGET" ]; then
    echo "Refusing to replace existing config: $TARGET" >&2
    echo "Back it up and explicitly approve replacement first." >&2
  else
    cp "$SOURCE" "$TARGET"
    echo "Installed: $TARGET"
  fi
fi
```

If a matching file **already exists**, make a backup and deliberately approve the replacement. The following commands are only for that case, after the variables above have been set:

```bash
BACKUP_DIR=$(mktemp -d "$PWD/tuned_configs/install-backup.XXXXXX")
cp -p "$TARGET" "$BACKUP_DIR/"
echo "Original saved in: $BACKUP_DIR"
cp -i "$SOURCE" "$TARGET"   # Confirm interactively before replacing.

# After validation, restore the original file:
cp -p "$BACKUP_DIR/$(basename "$TARGET")" "$TARGET"
```

If the target file **did not exist before the test**, rollback means removing **only the file you installed** (`rm -- "$TARGET"`). Repeat this process per generated shape; never copy the entire output directory or `reports/` into vLLM. Restart affected serving processes after either installation or rollback. File installation may require write permission to your vLLM environment.

In a **fresh Python process** after installation, verify the installed loader and public Triton wrapper with the **same shape, `M`, block layout, and output dtype** as your generated config:

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --save-path ./tuned_configs/quickstart \
  --verify-installed
```

This confirms loader contents and a numerical check of the public wrapper. It does **not** confirm that the full serving engine selects `TritonFp8BlockScaledMMKernel`: other backends such as CUTLASS, DeepGEMM, or FlashInfer may be selected instead. Verify the selected backend and actual model-serving behavior separately. The loader caches configurations, so restart serving processes after an intentional installation or rollback.

For full, reproducible hardware acceptance—including both FP16 and BF16 smoke tests, the model preview, and single-call vs repeated-call comparisons—see [CUDA validation](docs/CUDA_VALIDATION.md).

## Measurement methodology

Each `(M, N, K)` is tuned against the **installed vLLM private Triton kernel**, not a copied kernel implementation. The tuner:

1. Generates synthetic FP8 E4M3FN activations and weights with FP32 scales; all candidates for a shape use the same input tensors.
2. Warms up each candidate and measures CUDA-event latency. The default is **one GEMM call per event**; explicit `--calls-per-event >1` reports a repeated-call average, a different workload that can change cache behavior and rankings.
3. Shortlists the three fastest successful candidates, plus the default configuration if runnable, and remeasures them independently using medians.
4. Compares finalists to an FP32 dequantized matmul computed from the **same FP8 tensors and scales** (`rtol=0.02`, `atol=0.02`, with non-finite outputs rejected).
5. Saves the best passing finalist and a measurement report. Known Triton resource-exhaustion candidates are skipped; unexpected runtime/compilation errors abort rather than being hidden.

This is a **kernel-level sanity and tuning check**, not proof of end-to-end accuracy, representative production input distributions, global optimality, or a serving throughput improvement. The selected kernel configuration matters only if the deployed vLLM build actually uses this Triton path.

## Command reference

| Option | Purpose |
| --- | --- |
| `--model MODEL` / `--shape N K` | Mutually exclusive shape sources; `--shape` can be repeated |
| `--tp-size TP` | Target vLLM tensor-parallel size; default `1` |
| `--out-dtype auto\|float16\|bfloat16\|float32\|half` | `auto` for model mode; explicit in shape mode; `half` aliases `float16` |
| `--block-n N`, `--block-k K` | FP8 quantization block layout; defaults `128`, `128` |
| `--batch-size M` | Tune one flattened GEMM `M`; otherwise use the default 18-point grid |
| `--save-path DIR` | Write config JSON and reports under this directory; default `./tuned_configs` |
| `--overwrite` | Allow replacing overlapping `M` entries only |
| `--preview` | Print the task plan without running GPU kernels |
| `--check-environment` | Validate CUDA/vLLM/Triton compatibility and device capability |
| `--verify-installed` | Compare installed loader result with saved configs and test its public wrapper |
| `--measurements N`, `--calls-per-event N` | CUDA-event measurement rounds (default `5`) and calls per event (default `1`) |
| `--trust-remote-code` | Explicitly allow model repository code; **off by default** |

The wrapper scripts also recognize `BATCH_SIZE`, `OUT_DTYPE`, `SAVE_PATH`, `PYTHON`, and `TRUST_REMOTE_CODE=1`. For batch examples, `SAVE_PATH` or `--save-path` specifies the **batch root**; each model/TP task is written to its own subdirectory. See [`examples/tune_qwen3_models.sh`](examples/tune_qwen3_models.sh).

## Tests and troubleshooting

```bash
python3 -m pytest -q
# On a compatible CUDA machine, also run:
python3 -m pytest -q tests/test_gpu.py
```

The reviewed PR's Python 3.10/3.13 CPU/Shell checks passed, but the GPU module was skipped where CUDA/vLLM were unavailable. **A skipped GPU test is not a passed hardware test.** Detailed evidence is in [Validation notes](docs/VALIDATION.md).

| Symptom | Check |
| --- | --- |
| `--shape` asks for `--out-dtype` | Set the output dtype used by the intended serving workload; it cannot be inferred from dimensions alone. |
| Model architecture or exclusion rejected | The adapter cannot prove that a regular FP8 linear is present. Use a directly observed per-rank `--shape` only after checking the actual backend/layout. |
| Existing `M` conflicts | Choose a new output directory, request disjoint `M` values, or use `--overwrite` deliberately. |
| Kernel/helper/signature check fails | The installed vLLM version may be incompatible. Check the target build instead of silently falling back to another kernel. |
| Generated JSON does not affect serving | Check the exact filename, `M` selection, installed directory, process restart, and whether serving actually selects the Triton backend. |

## Project status and license

The actively maintained path is the CUDA W8A8 block-FP8 tuner. This repository no longer provides the legacy INT8/AWQ tuning implementations or their four Python entry points. External calls or imports using those removed entries will stop working. Their [historical source and AWQ documentation](https://github.com/massif-01/vLLM_benchmark_Block_FP8_Tuner/tree/392a13ad7e8b1e34b3e9e44576c957e841174697) remain available at the fixed commit before removal.

The FP8 compatibility entry [`benchmark_w8a8_block_fp8_qwencoder.py`](benchmark_w8a8_block_fp8_qwencoder.py) and DeepSeek migration-hint script [`scripts/tune_deepseek_v3.sh`](scripts/tune_deepseek_v3.sh) remain. The latter provides migration guidance, not DeepSeek automatic shape planning.

Source compatibility was reviewed against a fixed vLLM upstream revision, but this is **not** a guarantee for every vLLM release or a replacement for CUDA validation. See [Validation notes](docs/VALIDATION.md) for the exact revision and outstanding gates.

Distributed under the [Apache License 2.0](LICENSE). Original code provenance and attribution are documented in [NOTICE](NOTICE) and source headers.
