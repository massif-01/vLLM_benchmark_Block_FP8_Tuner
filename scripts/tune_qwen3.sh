#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/tune_custom.sh" "${1:-Qwen/Qwen3-8B}" "${2:-1}" \
    "${3:-128}" "${4:-128}" "${@:5}"
