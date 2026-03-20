#!/usr/bin/env bash
# Run LLaDA 1.5 with DUS (Dilated Unmasking Scheduler).
#
# Sweeps block_length = 1, 2, 4, 8, 16, 32 with dilated scheduler (base=2)
# and low_confidence remasking (threshold=0.3).
#
# Usage:
#   bash scripts/llada/llada-1.5/dus.sh                    # single GPU
#   bash scripts/llada/llada-1.5/dus.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for BL in 1 2 4 8 16 32; do
    echo ""
    echo "============================================"
    echo "Running DUS with block_length=${BL}"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs unmasking=dus,block_length=${BL},dus_base=2,dus_remasking_threshold=0.3 \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
