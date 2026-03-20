#!/usr/bin/env bash
# Run DiffuCoder with left-to-right sequential decoding.
#
# Left-to-right always uses k=1 (one token per step, leftmost first).
#
# Usage:
#   bash scripts/dream/diffucoder/left_to_right.sh                    # single GPU
#   bash scripts/dream/diffucoder/left_to_right.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

echo ""
echo "============================================"
echo "Running left-to-right (k=1)"
echo "============================================"

uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
    --model parallelbench_dream \
    --model_args model_path=apple/DiffuCoder-7B-Instruct \
    --gen_kwargs k=1,unmasking=left_to_right \
    --tasks parallelbench_all \
    --include_path parallelbench/tasks \
    --batch_size 8 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path "$OUTPUT_DIR"
