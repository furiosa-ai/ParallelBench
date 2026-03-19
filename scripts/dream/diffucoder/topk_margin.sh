#!/usr/bin/env bash
# Run DiffuCoder with top-k margin across k values.
#
# Sweeps k = 1, 2, 4, 8, 16, 32 with unmasking=topk_margin.
#
# Usage:
#   bash scripts/dream/diffucoder/topk_margin.sh                    # single GPU
#   bash scripts/dream/diffucoder/topk_margin.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for K in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running k=${K} with top-k margin"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_dream \
        --model_args model_path=apple/DiffuCoder-7B-Instruct \
        --gen_kwargs k=${K},unmasking=topk_margin \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
