# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from sglang quantization/tuning_block_wise_kernel.py
# Deprecated compatibility entry; repository modifications delegate to the maintained tuner.
import sys
from benchmark_w8a8_block_fp8 import cli

if __name__ == '__main__':
    print('Deprecated entry: use benchmark_w8a8_block_fp8.py with --model or --shape N K.', file=sys.stderr)
    sys.exit(cli())
