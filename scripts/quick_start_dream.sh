#!/usr/bin/env bash
# Quick start: Run Dream-v0-Instruct-7B on all 17 ParallelBench tasks and analyze results.
#
# Uses k (tokens per step) to auto-derive steps and block_length per task.
# Each task has its own max_tokens (32 for waiting_line, 64 for others),
# and the system computes steps = max_tokens / k automatically.
#
# Usage:
#   bash scripts/quick_start_dream.sh                    # single GPU
#   bash scripts/quick_start_dream.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"

uv run accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
    --model parallelbench_dream \
    --model_args model_path=Dream-org/Dream-v0-Instruct-7B \
    --gen_kwargs k=1,unmasking=random \
    --tasks parallelbench_all \
    --include_path parallelbench/tasks \
    --limit 32 \
    --batch_size 8 \
    --log_samples \
    --output_path "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Results summary"
echo "============================================"
uv run pb analyze "$OUTPUT_DIR"
