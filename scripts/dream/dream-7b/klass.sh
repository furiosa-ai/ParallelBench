#!/usr/bin/env bash
# Run Dream 7B with KLASS (KL-Adaptive Stability Sampling).
#
# Sweeps conf_threshold = 0.7, 0.8, 0.9, 0.95 with fixed kl_threshold=0.01.
#
# Usage:
#   bash scripts/dream/dream-7b/klass.sh                    # single GPU
#   bash scripts/dream/dream-7b/klass.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for CONF in 0.7 0.8 0.9 0.95; do
    echo ""
    echo "============================================"
    echo "Running KLASS with conf_threshold=${CONF}"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_dream \
        --model_args model_path=Dream-org/Dream-v0-Instruct-7B \
        --gen_kwargs unmasking=klass,conf_threshold=${CONF},kl_threshold=0.01,kl_history_length=2 \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
