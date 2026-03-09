#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 on one task per category with --limit 5.
#
# This script demonstrates the basic evaluation workflow:
#   - model_args: model identity only (model_path)
#   - gen_kwargs: generation parameters (steps, block_length, remasking)
#   - max_tokens: enforced per task category in YAML (32 for waiting_line, 64 for others)
#
# Usage:
#   bash scripts/quick_start.sh                  # single GPU
#   bash scripts/quick_start.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
INCLUDE_PATH="parallelbench/tasks"
LIMIT=5
BATCH_SIZE=1

MODEL="parallelbench_llada"
MODEL_ARGS="model_path=GSAI-ML/LLaDA-1.5"
GEN_KWARGS="steps=32,block_length=2,remasking=low_confidence"

TASKS=(
    "parallel_bench_waiting_line_shuffle"
    "parallel_bench_text_writing_words_to_sentence_easy"
    "parallel_bench_puzzles_sudoku_n4"
)

for task in "${TASKS[@]}"; do
    echo "============================================"
    echo "Running: $task"
    echo "============================================"

    accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --gen_kwargs "$GEN_KWARGS" \
        --tasks "$task" \
        --include_path "$INCLUDE_PATH" \
        --limit "$LIMIT" \
        --batch_size "$BATCH_SIZE" \
    || echo "[WARN] $task failed with exit code $?"

    echo ""
done

echo "Done!"
