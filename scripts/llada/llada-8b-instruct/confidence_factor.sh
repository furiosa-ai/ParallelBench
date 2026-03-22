#!/usr/bin/env bash
# Run LLaDA 8B Instruct with confidence factor across alg_factor values.
#
# Sweeps alg_factor = 0.7, 1.0, 1.3, 1.6, 1.9 with unmasking=confidence_factor.
#
# Usage:
#   bash scripts/llada/llada-8b-instruct/confidence_factor.sh                    # single GPU
#   bash scripts/llada/llada-8b-instruct/confidence_factor.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for alg_factor in 0.7 1.0 1.3 1.6 1.9; do
    echo ""
    echo "============================================"
    echo "Running alg_factor=${alg_factor} with confidence factor"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-8B-Instruct \
        --gen_kwargs alg_factor=${alg_factor},unmasking=confidence_factor \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 8 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
