"""Metric wrapper utilities for bridging ParallelBench metrics to lm-eval aggregation format.

ParallelBench metrics have two return patterns:
  1. float — e.g. list_match_score(pred, ref) -> 0.0/1.0
  2. dict  — e.g. sentence_to_words_score(pred, ref) -> {"score": 0.8, "grammar_score": 1.0, ...}

lm-eval expects process_results() to return dict[str, float]. This module bridges
the two by wrapping metric functions and flattening dict results.
"""

from __future__ import annotations

from typing import Callable

from parallelbench.dataset.metrics import Metric, parallel_bench_metric_func_map


def compute_sample_metrics(
    metric_name: str,
    prediction: str,
    reference,
    metric_func: Callable | None = None,
) -> dict[str, float]:
    """Compute per-sample metrics and return a flat dict.

    Returns keys like "score", "score_strict", or sub-metric names from dict-returning metrics.
    All values are raw (not scaled to percentage — aggregation handles that).
    """
    if metric_func is None:
        metric_func = parallel_bench_metric_func_map[metric_name]
        if isinstance(metric_func, type) and issubclass(metric_func, Metric):
            metric_func = metric_func()

    result = metric_func(prediction, reference)

    if isinstance(result, (int, float)):
        strict_result = metric_func(prediction, reference, strict=True)
        return {"score": float(result), "score_strict": float(strict_result)}
    elif isinstance(result, dict):
        return {k: float(v) for k, v in result.items()}
    else:
        return {"score": float(result)}


def get_metric_keys_for_task(metric_name: str) -> list[str]:
    """Return the expected metric keys for a given metric function name.

    Used to pre-register aggregation functions.
    """
    float_metrics = {
        "list_match_score",
        "list_shuffle_score",
        "list_random_insert_score",
        "list_random_remove_score",
        "list_random_replace_score",
        "sentence_random_remove_score",
        "sentence_random_replace_score",
        "sentence_random_insert_score",
        "latin_square_score",
        "math_op_score",
        "domino_score",
        "text_to_regex_score",
        "json_syntax_score",
        "sudoku_score",
    }

    dict_metrics = {
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

    if metric_name in float_metrics:
        return ["score", "score_strict"]
    elif metric_name in dict_metrics:
        return dict_metrics[metric_name]
    else:
        return ["score"]


# dLLM metadata metric keys appended to every task result
METADATA_METRIC_KEYS = [
    "nfe",
    "input_length",
    "output_length",
    "dec_order_kendall",
    "dec_order_spearman",
    "dec_order_kendall_ignore_pad",
    "dec_order_spearman_ignore_pad",
]
