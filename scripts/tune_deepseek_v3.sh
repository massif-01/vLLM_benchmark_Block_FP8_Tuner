#!/usr/bin/env bash
set -euo pipefail
echo 'Deprecated: DeepSeek-V3 automatic shapes were unverified and are no longer provided.' >&2
echo 'Use benchmark_w8a8_block_fp8.py --shape N K --out-dtype float16 (or the target runtime dtype) with observed per-rank CUDA FP8 regular-linear shapes.' >&2
exit 2
