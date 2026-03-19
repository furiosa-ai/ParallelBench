#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 with confidence topk strategy across k values.
#
# Sweeps k = 1, 2, 4, 8, 16, 32 with unmasking=confidence_topk (confidence topk).
# Each run uses --limit 32 samples per task.
#
# Usage:
#   bash scripts/confidence_topk/waiting_line.sh                    # single GPU
#   bash scripts/confidence_topk/waiting_line.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_conf_topk}"

for K in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running k=${K} with confidence topk"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs k=${K},unmasking=confidence_topk \
        --tasks parallelbench_waiting_line \
        --include_path parallelbench/tasks \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
