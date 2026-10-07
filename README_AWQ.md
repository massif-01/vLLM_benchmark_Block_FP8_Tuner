# AWQ (W4A16) Triton Kernel Tuning Script

> **Legacy / experimental / currently not maintained.** This document describes historical code, outside the maintained CUDA W8A8 Block FP8 workflow. Compatibility, numerical correctness and production consumption have not been validated.

## Description

This script (`benchmark_awq_w4a16.py`) is designed to tune the AWQ (Activation-aware Weight Quantization) Triton kernel for W4A16 quantization (4-bit weights, 16-bit activations) in vLLM. It performs automated kernel parameter search to find optimal configurations for different weight shapes and batch sizes.

## Features

- **AWQ W4A16 Quantization**: Tunes kernels for 4-bit weight quantization with 16-bit activations (FP16/BF16)
- **Group-wise Quantization**: Supports group sizes of 32, 64, 128, or -1 (per-channel)
- **Multi-GPU Support**: Automatically distributes tuning workload across available GPUs
- **Comprehensive Search Space**: Explores various block sizes, split-K configurations, and warp counts
- **Model-specific**: Currently configured for Qwen3 models (30B variant)

## Prerequisites

- CUDA-capable GPU (compute capability >= 7.5 for AWQ)
- vLLM installed with AWQ support
- Python 3.8+
- PyTorch with CUDA support

## Usage

### Basic Usage

```bash
python3 benchmark_awq_w4a16.py \
    --tp-size 1 \
    --group-size 128 \
    --split-k-iters 1 \
    --out-dtype float16 \
    --save-path ./awq_configs
```

### Parameters

- `--tp-size`: Tensor parallel size (default: 1)
- `--group-size`: AWQ group size for quantization (32, 64, 128, or -1 for per-channel, default: 128)
- `--split-k-iters`: Split-K iterations for parallelization (power of 2, default: 1)
- `--out-dtype`: Output data type (`float16` or `bfloat16`, default: `float16`)
- `--block-size-m`: Block size M (default: 32, used as initial value)
- `--block-size-n`: Block size N (default: 32, used as initial value)
- `--block-size-k`: Block size K (default: 32, used as initial value)
- `--batch-size`: Single batch size to tune (optional, if not specified, tunes multiple batch sizes)
- `--save-path`: Path to save configuration files (default: `./`)

### Advanced Usage

Tune for a specific batch size:
```bash
python3 benchmark_awq_w4a16.py \
    --tp-size 1 \
    --group-size 128 \
    --split-k-iters 1 \
    --batch-size 512 \
    --save-path ./awq_configs
```

Tune with tensor parallelism:
```bash
python3 benchmark_awq_w4a16.py \
    --tp-size 2 \
    --group-size 64 \
    --split-k-iters 2 \
    --save-path ./awq_configs_tp2
```

## Output Files

The script generates JSON configuration files for each weight shape (N, K) combination. The file naming format is:

```
N={N},K={K},G={group_size},device_name={device_name},dtype=awq_w4a16.json
```

Example:
```
N=2048,K=4096,G=128,device_name=NVIDIA_A100,dtype=awq_w4a16.json
```

Each configuration file contains optimal kernel parameters for different batch sizes (M dimensions), with keys like:
- `BLOCK_SIZE_M`: Block size for M dimension
- `BLOCK_SIZE_N`: Block size for N dimension
- `BLOCK_SIZE_K`: Block size for K dimension
- `SPLIT_K`: Split-K iterations
- `num_warps`: Number of warps
- `num_stages`: Number of pipeline stages

## Weight Shapes

The script is currently configured for Qwen3-30B-A3B-Instruct models with the following weight shapes (for tp_size=1):

- `(1536, 2048)` - MoE gate_up_proj (merged)
- `(5120, 2048)` - Attention QKV projection
- `(2048, 4096)` - Attention O projection
- `(2048, 768)` - MoE down_proj

To modify for other models, update the `get_weight_shapes()` function in the script.

## How It Works

1. **Weight Quantization**: The script creates synthetic FP16/BF16 weights and quantizes them to INT4 using AWQ format:
   - Quantizes weights to 4-bit integers (range: -8 to 7, stored as 0-15)
   - Packs 2 4-bit values into 1 uint8
   - Computes group-wise scales and zeros

2. **Kernel Tuning**: For each weight shape and batch size:
   - Explores a comprehensive search space of kernel configurations
   - Benchmarks each configuration
   - Selects the configuration with the lowest latency

3. **Multi-GPU Parallelization**: Distributes batch sizes across available GPUs to speed up tuning

## Integration with vLLM

The historical AWQ script writes a custom JSON format. The source-audited vLLM AWQ runtime does not load these files automatically. Copying them into the FP8 config directory does not enable AWQ tuning results. A production consumer would need separate work, which is outside this repair scope.

## Notes

- AWQ quantization requires weights to be quantized in a specific format (packed INT4)
- The script supports group sizes: 32, 64, 128, or -1 (per-channel)
- Split-K iterations must be a power of 2 and <= 32
- Tuning may take several hours depending on the number of weight shapes and batch sizes

## Troubleshooting

**Issue**: `ImportError: cannot import name 'awq_gemm_triton'`
- **Solution**: Ensure vLLM is properly installed with AWQ support

**Issue**: `RuntimeError: No GPU available`
- **Solution**: Check CUDA installation and GPU availability

**Issue**: Configuration search fails for certain shapes
- **Solution**: Some weight shapes may not be compatible with certain block sizes. The script will skip invalid configurations automatically.

## References

- AWQ Paper: [Activation-aware Weight Quantization for Neural Network Compression](https://arxiv.org/abs/2306.00978)
- vLLM AWQ Documentation: See `docs/features/quantization/auto_awq.md`
- AWQ Triton Kernel: `vllm/model_executor/layers/quantization/awq_triton.py`

