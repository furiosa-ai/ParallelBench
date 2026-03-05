#!/usr/bin/env bash
# Run all models on all ParallelBench tasks with --limit 4.
# max_tokens=32 for waiting_line tasks, max_tokens=64 for the rest.
# Usage: bash scripts/run_all.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-results}"
INCLUDE_PATH="parallelbench/lm_eval_tasks"
LIMIT=4
BATCH_SIZE=1

# Tasks grouped by max_tokens
WAITING_LINE_TASKS=$(echo parallel_bench_waiting_line_{copy,reverse,sort,shuffle,insert_index,insert_random,remove_index,remove_random,replace_index,replace_random} | tr ' ' ',')
WAITING_LINE_N15_TASKS=$(echo parallel_bench_waiting_line_n15_{copy,reverse,sort,shuffle,insert_index,insert_random,remove_index,remove_random,replace_index,replace_random} | tr ' ' ',')
WORDS_TASKS=$(echo parallel_bench_text_writing_words_to_sentence_{easy,easy_n{1,3,4,5,6,7},medium,medium_n{1,3,4,5,6,7},hard,hard_n{1,3,4,5,6,7}} | tr ' ' ',')
TEXT_WRITING_TASKS="parallel_bench_text_writing_paraphrasing,parallel_bench_text_writing_summarization"
PUZZLE_TASKS="parallel_bench_puzzles_sudoku_n4_12,parallel_bench_puzzles_latin_square_n4"

# max_tokens=32
TASKS_32="${WAITING_LINE_TASKS},${WAITING_LINE_N15_TASKS}"
# max_tokens=64
TASKS_64="${WORDS_TASKS},${TEXT_WRITING_TASKS},${PUZZLE_TASKS}"

TASK_GROUPS=(
    "${TASKS_32}:32"
    "${TASKS_64}:64"
)

# model_name:wrapper:model_path:extra_args
DLLM_MODELS=(
    "llada_1.5:parallelbench_llada:GSAI-ML/LLaDA-1.5:remasking=low_confidence"
    "llada_8b:parallelbench_llada:GSAI-ML/LLaDA-8B-Instruct:remasking=low_confidence"
    "dream:parallelbench_dream:Dream-org/Dream-v0-Instruct-7B:remasking=origin"
    "dream_coder:parallelbench_dream:Dream-org/Dream-Coder-v0-Instruct-7B:remasking=origin"
    "diffucoder:parallelbench_dream:apple/DiffuCoder-7B-Instruct:remasking=origin"
    "diffucoder_cpgrpo:parallelbench_dream:apple/DiffuCoder-7B-cpGRPO:remasking=origin"
    "trado_4b:parallelbench_trado:Gen-Verse/TraDo-4B-Instruct:remasking=random"
    "trado_8b:parallelbench_trado:Gen-Verse/TraDo-8B-Instruct:remasking=random"
    "trado_8b_thinking:parallelbench_trado:Gen-Verse/TraDo-8B-Thinking:remasking=random"
)

API_MODELS=(
    "haiku:parallelbench_api:claude-3-haiku"
    "mercury:parallelbench_api:mercury-coder"
)

TOTAL_MODELS=$(( ${#DLLM_MODELS[@]} + ${#API_MODELS[@]} ))
TOTAL_RUNS=$(( TOTAL_MODELS * ${#TASK_GROUPS[@]} ))
CURRENT=0

for task_entry in "${TASK_GROUPS[@]}"; do
    TASKS="${task_entry%%:*}"
    MAX_TOKENS="${task_entry##*:}"

    for model_entry in "${DLLM_MODELS[@]}"; do
        IFS=: read -r name wrapper model_path extra <<< "$model_entry"
        CURRENT=$((CURRENT + 1))
        echo "============================================"
        echo "[$CURRENT/$TOTAL_RUNS] $name | max_tokens=$MAX_TOKENS"
        echo "============================================"

        uv run accelerate launch --main_process_port 0 -m parallelbench.cli \
            --model "$wrapper" \
            --model_args "model_path=$model_path,steps=$MAX_TOKENS,max_tokens=$MAX_TOKENS,block_length=$MAX_TOKENS,$extra" \
            --tasks "$TASKS" \
            --include_path "$INCLUDE_PATH" \
            --limit "$LIMIT" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --output_path "$OUTPUT_DIR/$name" \
        || echo "[WARN] $name / max_tokens=$MAX_TOKENS failed with exit code $?"
    done

    for model_entry in "${API_MODELS[@]}"; do
        IFS=: read -r name wrapper model_path <<< "$model_entry"
        CURRENT=$((CURRENT + 1))
        echo "============================================"
        echo "[$CURRENT/$TOTAL_RUNS] $name | max_tokens=$MAX_TOKENS"
        echo "============================================"

        uv run accelerate launch --main_process_port 0 -m parallelbench.cli \
            --model "$wrapper" \
            --model_args "model_path=$model_path,max_tokens=$MAX_TOKENS" \
            --tasks "$TASKS" \
            --include_path "$INCLUDE_PATH" \
            --limit "$LIMIT" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --output_path "$OUTPUT_DIR/$name" \
        || echo "[WARN] $name / max_tokens=$MAX_TOKENS failed with exit code $?"
    done
done

echo ""
echo "All $TOTAL_RUNS runs complete. Results saved to $OUTPUT_DIR/"
