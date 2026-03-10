"""Metric wrapper utilities for bridging ParallelBench metrics to lm-eval aggregation format.

All ParallelBench metric functions return dict[str, float].
lm-eval expects process_results() to return dict[str, float].
"""

from __future__ import annotations

from typing import Callable

from parallelbench.datasets.metrics import Metric, parallel_bench_metric_func_map


def compute_sample_metrics(
    metric_name: str,
    prediction: str,
    reference,
    metric_func: Callable | None = None,
) -> dict[str, float]:
    """Compute per-sample metrics and return a flat dict.

    All values are raw (not scaled to percentage — aggregation handles that).
    """
    if metric_func is None:
        metric_func = parallel_bench_metric_func_map[metric_name]
        if isinstance(metric_func, type) and issubclass(metric_func, Metric):
            metric_func = metric_func()

    return metric_func(prediction, reference)


# Unified metric key registry — all metrics return dict[str, float]
METRIC_KEYS: dict[str, list[str]] = {
    "list_match_score": ["score", "score_strict"],
    "list_shuffle_score": ["score", "score_strict"],
    "list_random_insert_score": ["score", "score_strict"],
    "list_random_remove_score": ["score", "score_strict"],
    "list_random_replace_score": ["score", "score_strict"],
    "sentence_random_remove_score": ["score", "score_strict"],
    "sentence_random_replace_score": ["score", "score_strict"],
    "sentence_random_insert_score": ["score", "score_strict"],
    "latin_square_score": ["score", "score_strict"],
    "math_op_score": ["score", "score_strict"],
    "domino_score": ["score", "score_strict"],
    "text_to_regex_score": ["score", "score_strict"],
    "json_syntax_score": ["score", "score_strict"],
    "sudoku_score": ["score", "score_strict"],
    "sentence_replace_all_with_unique_random_score": ["score", "score_loose"],
    "sentence_to_words_score": ["inclusion_score", "grammar_score", "score"],
    "grammar_score": ["score"],
    "startwith_score": ["grammar_score", "startswith_score", "score"],
    "regex_match_score": ["score"],
    "summary_score": [
        "rouge1_score",
        "rouge2_score",
        "rougeL_score",
        "grammar_score",
        "score",
    ],
    "paraphrase_score": [
        "inv_bleu_score",
        "bertscore_score",
        "grammar_score",
        "score",
    ],
}


def get_metric_keys_for_task(metric_name: str) -> list[str]:
    """Return the expected metric keys for a given metric function name."""
    return METRIC_KEYS.get(metric_name, ["score"])


# dLLM metadata metric keys appended to every task result
METADATA_METRIC_KEYS = [
    "nfe",
    "tokens_per_step",
    "input_length",
    "output_length",
    "dec_order_kendall",
    "dec_order_spearman",
    "dec_order_kendall_ignore_pad",
    "dec_order_spearman_ignore_pad",
]
