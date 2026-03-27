#!/usr/bin/env bash
# Run TraDo-8B-Instruct with confidence threshold across alg_threshold values.
#
# Sweeps alg_threshold = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 with block_length=4.
#
# Usage:
#   bash scripts/trado/trado-8b-instruct/confidence_threshold.sh                    # single GPU
#   bash scripts/trado/trado-8b-instruct/confidence_threshold.sh --num_processes 2  # multi GPU

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
        --model parallelbench_trado \
        --model_args model_path=Gen-Verse/TraDo-8B-Instruct \
        --gen_kwargs alg_threshold=${alg_threshold},unmasking=confidence_threshold \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
