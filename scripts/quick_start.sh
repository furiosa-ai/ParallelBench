#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 on all 17 ParallelBench tasks and analyze results.
#
# Uses k (tokens per step) to auto-derive steps and block_length per task.
# Each task has its own max_tokens (32 for waiting_line, 64 for others),
# and the system computes steps = max_tokens / k automatically.
#
# Usage:
#   bash scripts/quick_start.sh                    # single GPU
#   bash scripts/quick_start.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"

uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
    --model parallelbench_llada \
    --model_args model_path=GSAI-ML/LLaDA-1.5 \
    --gen_kwargs k=1,remasking=random \
    --tasks parallelbench \
    --include_path parallelbench/tasks \
    --limit 2 \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Results summary"
echo "============================================"
uv run pb analyze "$OUTPUT_DIR"
