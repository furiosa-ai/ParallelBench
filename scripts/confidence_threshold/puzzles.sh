#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 with confidence threshold strategy across alg_threshold values.
#
# Sweeps alg_threshold = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 with unmasking=confidence_threshold (confidence threshold).

#
# Usage:
#   bash scripts/confidence_threshold/puzzles.sh                    # single GPU
#   bash scripts/confidence_threshold/puzzles.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_puzzles}"

for alg_threshold in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo ""
    echo "============================================"
    echo "Running alg_threshold=${alg_threshold} with confidence threshold"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs alg_threshold=${alg_threshold},unmasking=confidence_threshold \
        --tasks parallelbench_puzzles \
        --include_path parallelbench/tasks \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
