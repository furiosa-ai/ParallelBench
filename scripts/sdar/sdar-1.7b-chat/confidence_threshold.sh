#!/usr/bin/env bash
# Run SDAR-1.7B-Chat with confidence threshold across alg_threshold values.
#
# Sweeps alg_threshold = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 with unmasking=confidence_threshold.
#
# Usage:
#   bash scripts/sdar/sdar-1.7b-chat/confidence_threshold.sh                    # single GPU
#   bash scripts/sdar/sdar-1.7b-chat/confidence_threshold.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for alg_threshold in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo ""
    echo "============================================"
    echo "Running alg_threshold=${alg_threshold} with confidence threshold"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_sdar \
        --model_args model_path=JetLM/SDAR-1.7B-Chat \
        --gen_kwargs alg_threshold=${alg_threshold},unmasking=confidence_threshold \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
