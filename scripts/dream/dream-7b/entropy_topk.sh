#!/usr/bin/env bash
# Run Dream 7B with entropy top-k across k values.
#
# Sweeps k = 1, 2, 4, 8, 16, 32 with unmasking=entropy_topk.
#
# Usage:
#   bash scripts/dream/dream-7b/entropy_topk.sh                    # single GPU
#   bash scripts/dream/dream-7b/entropy_topk.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for K in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running k=${K} with entropy top-k"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_dream \
        --model_args model_path=Dream-org/Dream-v0-Instruct-7B \
        --gen_kwargs k=${K},unmasking=entropy_topk \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
