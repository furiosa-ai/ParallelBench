#!/usr/bin/env bash
# Quick start: Run LLaDA-1.5 on all ParallelBench tasks with --limit 8.
#
# This script demonstrates the basic evaluation workflow:
#   - model_args: model identity only (model_path)
#   - gen_kwargs: generation parameters (steps, block_length, remasking)
#   - max_tokens: enforced per task category in YAML (32 for waiting_line, 64 for others)
#
# Here we set steps=max_tokens and block_length=max_tokens for fully parallel decoding.
#
# Usage:
#   bash scripts/quick_start.sh                    # single GPU
#   bash scripts/quick_start.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
INCLUDE_PATH="parallelbench/tasks"
OUTPUT_DIR="results"
LIMIT=8
BATCH_SIZE=1

MODEL="parallelbench_llada"
MODEL_ARGS="model_path=GSAI-ML/LLaDA-1.5"

GEN_KWARGS_32="steps=32,block_length=32,remasking=low_confidence"
GEN_KWARGS_64="steps=64,block_length=64,remasking=low_confidence"

# task:gen_kwargs
RUNS=(
    # Waiting Line (max_tokens=32)
    "parallel_bench_waiting_line_copy:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_reverse:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_sort:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_shuffle:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_insert_index:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_insert_random:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_remove_index:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_remove_random:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_replace_index:${GEN_KWARGS_32}"
    "parallel_bench_waiting_line_replace_random:${GEN_KWARGS_32}"
    # Text Writing (max_tokens=64)
    "parallel_bench_text_writing_paraphrasing:${GEN_KWARGS_64}"
    "parallel_bench_text_writing_summarization:${GEN_KWARGS_64}"
    "parallel_bench_text_writing_words_to_sentence_easy:${GEN_KWARGS_64}"
    "parallel_bench_text_writing_words_to_sentence_medium:${GEN_KWARGS_64}"
    "parallel_bench_text_writing_words_to_sentence_hard:${GEN_KWARGS_64}"
    # Puzzles (max_tokens=64)
    "parallel_bench_puzzles_sudoku_n4:${GEN_KWARGS_64}"
    "parallel_bench_puzzles_latin_square_n4:${GEN_KWARGS_64}"
)

TOTAL=${#RUNS[@]}
CURRENT=0

for entry in "${RUNS[@]}"; do
    task="${entry%%:*}"
    gen_kwargs="${entry#*:}"
    CURRENT=$((CURRENT + 1))

    echo "============================================"
    echo "[$CURRENT/$TOTAL] $task"
    echo "  gen_kwargs: $gen_kwargs"
    echo "============================================"

    accelerate launch ${EXTRA_ARGS} -m parallelbench.cli.eval \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --gen_kwargs "$gen_kwargs" \
        --tasks "$task" \
        --include_path "$INCLUDE_PATH" \
        --limit "$LIMIT" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --output_path "$OUTPUT_DIR" \
    || echo "[WARN] $task failed with exit code $?"

    echo ""
done

echo "Done! $TOTAL tasks complete."
