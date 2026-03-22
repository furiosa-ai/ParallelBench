#!/usr/bin/env bash
# Run SDAR-1.7B-Chat with random unmasking across k values.
#
# Sweeps k = 1, 2, 4, 8, 16, 32 with unmasking=random.
#
# Usage:
#   bash scripts/sdar/sdar-1.7b-chat/random.sh                    # single GPU
#   bash scripts/sdar/sdar-1.7b-chat/random.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for K in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running k=${K} with random"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_sdar \
        --model_args model_path=JetLM/SDAR-1.7B-Chat \
        --gen_kwargs k=${K},unmasking=random \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
