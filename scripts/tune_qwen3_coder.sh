#!/usr/bin/env bash
# Compatibility convenience entry; no model-specific optimized search space.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/tune_custom.sh" "${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}" \
    "${2:-4}" "${3:-128}" "${4:-128}" "${@:5}"
