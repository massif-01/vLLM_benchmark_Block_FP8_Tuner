# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source fixture: vLLM c741bfca70cfb777e2016f827eae31f6e215fe9f (2026-10-08).
# Official helper/loader bodies only, for CPU contract tests; not a runtime fallback.
import functools
import json
import os
import re
from typing import Any

def get_device_name_as_file_name(device_id: int = 0) -> str:
    from vllm.platforms import current_platform

    name = current_platform.get_device_name(device_id)
    name = re.sub(r"[\s/]+", "_", name)
    return name

def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    """Return optimized configurations for the w8a8 block fp8 kernel.
    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = get_device_name_as_file_name()
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json"  # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block FP8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None

KERNEL_ARG_NAMES = ['A', 'B', 'C', 'As', 'Bs', 'M', 'N', 'K', 'group_n', 'group_k', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn', 'stride_cm', 'stride_cn', 'stride_As_m', 'stride_As_k', 'stride_Bs_k', 'stride_Bs_n', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M']

def default_config(block_size):
    return {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_size[0],
                "BLOCK_SIZE_K": block_size[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2,
            }
