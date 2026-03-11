#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 with confidence topk strategy across k values.
#
# Sweeps k = 1, 2, 4, 8, 16, 32 with remasking=confidence_topk (confidence topk).
# Each run uses --limit 32 samples per task.
#
# Usage:
#   bash scripts/quick_start_confidence_topk.sh                    # single GPU
#   bash scripts/quick_start_confidence_topk.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"

for K in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running k=${K} with confidence topk"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs k=${K},remasking=confidence_topk \
        --tasks parallel_bench \
        --include_path parallelbench/tasks \
        --limit 32 \
        --batch_size 1 \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done

echo ""
echo "============================================"
echo "Results summary"
echo "============================================"
uv run pb analyze "$OUTPUT_DIR"
