"""Mapping between ParallelBench internal names and GitHub Pages IDs.

Provides bidirectional lookup dicts and helper functions for converting:
- model_name (results JSON) -> GitHub Pages model_id
- unmasking method name -> GitHub Pages strategy_id
- task name (lm-eval format) -> GitHub Pages task_id
"""

from __future__ import annotations

# Maps model_name as stored in results JSON to GitHub Pages model_id.
# Results directories use '__' instead of '/' in folder names
# (e.g., GSAI-ML__LLaDA-1.5), but the JSON inside uses '/'.
MODEL_ID_MAP: dict[str, str] = {
    "GSAI-ML/LLaDA-1.5": "llada15",
    "GSAI-ML/LLaDA-8B-Instruct": "llada10",
    "Dream-org/Dream-v0-Instruct-7B": "dream",
    "apple/DiffuCoder-7B-Instruct": "diffucoder",
    "GSAI-ML/LLaDA-MoE-7B-Instruct": "llada-moe",
    "GSAI-ML/LLaDA-MoE-7B-Instruct-TD": "llada-moe-td",
    "inclusionAI/LLaDA-2.0-Mini": "llada20-mini",
    "inclusionAI/LLaDA-2.0-Mini-CAP": "llada20-mini-cap",
    "inclusionAI/LLaDA-2.1-Mini": "llada21-mini",
    "NVlabs/dParallel-Dream-7B": "dparallel-dream",
    "NVlabs/dParallel-LLaDA-8B": "dparallel-llada",
    "JetAstra/SDAR-1.7B": "sdar-1.7b",
    "JetAstra/SDAR-4B": "sdar-4b",
    "JetAstra/SDAR-8B": "sdar-8b",
    "Gen-Verse/SDAR-TraDo-4B": "sdar-trado-4b",
    "Gen-Verse/SDAR-TraDo-8B": "sdar-trado-8b",
}

# Maps internal unmasking method name to GitHub Pages strategy_id.
# Includes all methods from UNMASKING_REGISTRY plus extra strategies
# defined in the GitHub Pages metadata.json.
STRATEGY_ID_MAP: dict[str, str] = {
    # Methods from UNMASKING_REGISTRY
    "random": "random",
    "left_to_right": "l2r",
    "confidence_threshold": "confidence-threshold",
    "confidence_topk": "confidence-topk",
    "confidence_factor": "confidence-factor",
    "entropy_topk": "entropy-topk",
    "topk_margin": "topk-margin",
    "klass": "klass",
    "slowfast": "slowfast",
    "dus": "dus",
    "wino_dllm": "wino",
    "origin": "random",  # origin is sequential baseline, maps to random
    # Extra strategies from GitHub Pages metadata.json not in UNMASKING_REGISTRY
    "apd": "apd",
    "confidence_eb": "confidence-eb",
    "confidence_pc_sampler": "confidence-pc-sampler",
    "random_pc_sampler": "random-pc-sampler",
    "confidence_threshold_quality": "confidence-threshold-quality",
    "confidence_threshold_speed": "confidence-threshold-speed",
}

# Maps internal task name (lm-eval format) to GitHub Pages task_id.
# Pattern: remove 'parallelbench_' prefix, replace 'waiting_line' with
# 'waitingline', replace 'text_writing' with 'textwriting', join category
# and task with '-'. Puzzles use special display names (sudoku, latin_square).
TASK_ID_MAP: dict[str, str] = {
    # Waiting Line tasks (10)
    "parallelbench_waiting_line_copy": "waitingline-copy",
    "parallelbench_waiting_line_reverse": "waitingline-reverse",
    "parallelbench_waiting_line_shuffle": "waitingline-shuffle",
    "parallelbench_waiting_line_sort": "waitingline-sort",
    "parallelbench_waiting_line_insert_index": "waitingline-insert_index",
    "parallelbench_waiting_line_insert_random": "waitingline-insert_random",
    "parallelbench_waiting_line_remove_index": "waitingline-remove_index",
    "parallelbench_waiting_line_remove_random": "waitingline-remove_random",
    "parallelbench_waiting_line_replace_index": "waitingline-replace_index",
    "parallelbench_waiting_line_replace_random": "waitingline-replace_random",
    # Text Writing tasks (5)
    "parallelbench_text_writing_summarization": "textwriting-summarization",
    "parallelbench_text_writing_paraphrasing": "textwriting-paraphrasing",
    "parallelbench_text_writing_w2s_easy": "textwriting-w2s_easy",
    "parallelbench_text_writing_w2s_medium": "textwriting-w2s_medium",
    "parallelbench_text_writing_w2s_hard": "textwriting-w2s_hard",
    # Puzzle tasks (2)
    "parallelbench_puzzles_sudoku_n4_12": "puzzles-sudoku",
    "parallelbench_puzzles_latin_square_n5": "puzzles-latin_square",
    # Legacy task names from older result files
    "parallelbench_puzzles_sudoku_n4": "puzzles-sudoku",
    "parallelbench_puzzles_latin_square_n4": "puzzles-latin_square",
    "parallelbench_text_writing_words_to_sentence_easy": "textwriting-w2s_easy",
    "parallelbench_text_writing_words_to_sentence_medium": "textwriting-w2s_medium",
    "parallelbench_text_writing_words_to_sentence_hard": "textwriting-w2s_hard",
}


def get_model_id(model_name: str) -> str | None:
    """Return GitHub Pages model_id for the given model_name, or None if unmapped."""
    return MODEL_ID_MAP.get(model_name)


def get_strategy_id(unmasking: str) -> str | None:
    """Return GitHub Pages strategy_id for the given unmasking method, or None if unmapped."""
    return STRATEGY_ID_MAP.get(unmasking)


def get_task_id(task_name: str) -> str | None:
    """Return GitHub Pages task_id for the given internal task name, or None if unmapped."""
    return TASK_ID_MAP.get(task_name)
