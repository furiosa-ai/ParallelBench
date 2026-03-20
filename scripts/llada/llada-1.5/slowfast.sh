#!/usr/bin/env bash
# Run LLaDA 1.5 with SlowFast Sampling.
#
# Sweeps sf_high_confidence_threshold = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
# with fixed cycle_len_confidence_threshold=0.3.
#
# Usage:
#   bash scripts/llada/llada-1.5/slowfast.sh                    # single GPU
#   bash scripts/llada/llada-1.5/slowfast.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for THRESH in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo ""
    echo "============================================"
    echo "Running SlowFast with sf_high_confidence_threshold=${THRESH}"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs unmasking=slowfast,sf_high_confidence_threshold=${THRESH},sf_cycle_confidence_threshold=0.3 \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
