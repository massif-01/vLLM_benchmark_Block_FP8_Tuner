#!/usr/bin/env bash
# Batch example: regular-linear shapes only, never routed MoE experts.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TUNE_SCRIPT="$SCRIPT_DIR/../scripts/tune_qwen3.sh"
# SAVE_PATH and --save-path select the batch root, not a shared config directory.
BASE_SAVE_PATH=${SAVE_PATH:-"$SCRIPT_DIR/../tuned_configs/batch"}
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-path)
            BASE_SAVE_PATH=${2:?--save-path requires a directory}
            shift 2 ;;
        --save-path=*) BASE_SAVE_PATH=${1#*=}; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-30B-A3B")
TP_SIZES=(1 2 4 8)
FAILURES=0
for model in "${MODELS[@]}"; do
    for tp in "${TP_SIZES[@]}"; do
        echo "Tuning: $model with target TP=$tp"
        TASK_SAVE_PATH="$BASE_SAVE_PATH/${model//\//_}/tp_$tp"
        TASK_ARGS=("$model" "$tp" 128 128)
        # Bash 3 with nounset cannot expand an empty array.
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            TASK_ARGS+=("${EXTRA_ARGS[@]}")
        fi
        if ! SAVE_PATH="$TASK_SAVE_PATH" bash "$TUNE_SCRIPT" "${TASK_ARGS[@]}"; then
            echo "Tuning failed: $model TP=$tp" >&2
            FAILURES=$((FAILURES + 1))
        fi
    done
done
if [[ "$FAILURES" -gt 0 ]]; then
    echo "Batch tuning failed: $FAILURES task(s) failed" >&2
    exit 1
fi
echo 'Batch tasks completed successfully'
