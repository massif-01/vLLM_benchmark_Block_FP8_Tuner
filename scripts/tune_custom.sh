#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL=${1:?Usage: tune_custom.sh MODEL [TP] [BLOCK_N] [BLOCK_K] [extra CLI flags]}
TP=${2:-1}
BLOCK_N=${3:-128}
BLOCK_K=${4:-128}
# Extra flags follow the four positional arguments, e.g. MODEL 4 128 128 --preview.
POSITIONALS=$(( $# < 4 ? $# : 4 ))
shift "$POSITIONALS"
SAVE_PATH=${SAVE_PATH:-"$PROJECT_DIR/tuned_configs"}
ARGS=(--model "$MODEL" --tp-size "$TP" --block-n "$BLOCK_N" --block-k "$BLOCK_K"
      --input-type "${INPUT_TYPE:-fp8}" --out-dtype "${OUT_DTYPE:-float16}" --save-path "$SAVE_PATH")
case "${TRUST_REMOTE_CODE:-0}" in
    0) ;;
    1) ARGS+=(--trust-remote-code) ;;
    *) echo 'TRUST_REMOTE_CODE must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -n "${BATCH_SIZE:-}" ]]; then
    ARGS+=(--batch-size "$BATCH_SIZE")
fi
"${PYTHON:-python3}" "$PROJECT_DIR/benchmark_w8a8_block_fp8.py" "${ARGS[@]}" "$@"
# Python itself reports tuning success. Help, preview and environment checks are not tuning.
