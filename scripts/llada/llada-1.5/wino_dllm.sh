#!/usr/bin/env bash
# Run LLaDA 1.5 with WINO-DLLM (Wide-In, Narrow-Out Revokable Decoding).
#
# Sweeps wino_threshold = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
# with fixed wino_threshold_back=0.9.
#
# Usage:
#   bash scripts/llada/llada-1.5/wino_dllm.sh                    # single GPU
#   bash scripts/llada/llada-1.5/wino_dllm.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

for THRESH in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo ""
    echo "============================================"
    echo "Running WINO-DLLM with wino_threshold=${THRESH}"
    echo "============================================"

    uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model parallelbench_llada \
        --model_args model_path=GSAI-ML/LLaDA-1.5 \
        --gen_kwargs unmasking=wino_dllm,wino_threshold=${THRESH},wino_threshold_back=0.9 \
        --tasks parallelbench_all \
        --include_path parallelbench/tasks \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --log_samples \
        --output_path "$OUTPUT_DIR"
done
