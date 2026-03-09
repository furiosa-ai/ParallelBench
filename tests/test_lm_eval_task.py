"""Tests for ParallelBenchTask and metric utilities."""

import importlib.util

import pytest

from parallelbench.tasks.utils import (
    METADATA_METRIC_KEYS,
    compute_sample_metrics,
    get_metric_keys_for_task,
)

_has_language_tool = importlib.util.find_spec("language_tool_python") is not None


class TestComputeSampleMetrics:
    """Test metric computation via compute_sample_metrics wrapper."""

    def test_list_match_score_correct(self):
        prediction = '["Alice", "Bob", "Charlie"]'
        reference = '["Alice", "Bob", "Charlie"]'
        result = compute_sample_metrics("list_match_score", prediction, reference)
        assert result["score"] == 1.0
        assert result["score_strict"] == 1.0

    def test_list_match_score_incorrect(self):
        prediction = '["Bob", "Alice", "Charlie"]'
        reference = '["Alice", "Bob", "Charlie"]'
        result = compute_sample_metrics("list_match_score", prediction, reference)
        assert result["score"] == 0.0

    def test_list_match_score_non_strict_format(self):
        prediction = "Alice, Bob, Charlie"
        reference = '["Alice", "Bob", "Charlie"]'
        result = compute_sample_metrics("list_match_score", prediction, reference)
        assert result["score"] == 1.0
        assert result["score_strict"] == 0.0

    @pytest.mark.skipif(
        not _has_language_tool,
        reason="language_tool_python not installed",
    )
    def test_dict_returning_metric(self):
        """sentence_to_words_score returns a dict with multiple keys."""
        prediction = "The quick brown fox jumps"
        reference = {"words": ["quick", "fox"]}
        result = compute_sample_metrics(
            "sentence_to_words_score", prediction, reference
        )
        assert "score" in result
        assert "inclusion_score" in result
        assert "grammar_score" in result
        assert result["inclusion_score"] == 1.0

    def test_sudoku_score_correct(self):
        prediction = "1234\n3412\n2143\n4321"
        reference = "1234\n3412\n2143\n4321"
        result = compute_sample_metrics("sudoku_score", prediction, reference)
        assert result["score"] == 1.0

    def test_sudoku_score_incorrect(self):
        prediction = "1234\n3412\n2143\n4322"
        reference = "1234\n3412\n2143\n4321"
        result = compute_sample_metrics("sudoku_score", prediction, reference)
        assert result["score"] == 0.0


class TestGetMetricKeysForTask:
    def test_float_metric_returns_score_and_strict(self):
        keys = get_metric_keys_for_task("list_match_score")
        assert keys == ["score", "score_strict"]

    def test_dict_metric_returns_all_keys(self):
        keys = get_metric_keys_for_task("sentence_to_words_score")
        assert "score" in keys
        assert "inclusion_score" in keys
        assert "grammar_score" in keys

    def test_summary_score_keys(self):
        keys = get_metric_keys_for_task("summary_score")
        assert "rouge1_score" in keys
        assert "rougeL_score" in keys
        assert "grammar_score" in keys
        assert "score" in keys

    def test_paraphrase_score_keys(self):
        keys = get_metric_keys_for_task("paraphrase_score")
        assert "inv_bleu_score" in keys
        assert "bertscore_score" in keys

    def test_unknown_metric_returns_score(self):
        keys = get_metric_keys_for_task("unknown_metric")
        assert keys == ["score"]


class TestMetadataMetricKeys:
    def test_contains_nfe(self):
        assert "nfe" in METADATA_METRIC_KEYS

    def test_contains_decoding_order_keys(self):
        assert "dec_order_kendall" in METADATA_METRIC_KEYS
        assert "dec_order_spearman" in METADATA_METRIC_KEYS
