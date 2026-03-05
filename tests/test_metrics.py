"""Tests for parallelbench.dataset.metrics functions.

Validates metric correctness and return type consistency.
All public metric functions must return dict[str, float].
"""

import pytest

from parallelbench.dataset.metrics import (
    _parse_list,
    list_match_score,
    domino_score,
    latin_square_score,
    math_op_score,
    text_to_regex_score,
    json_syntax_score,
    sudoku_score,
    sentence_to_words_score,
    startwith_score,
    regex_match_score,
    Metric,
    parallel_bench_metric_func_map,
)


class TestParseList:
    def test_basic_parse(self):
        assert _parse_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_non_strict_tolerant(self):
        result = _parse_list("a, b, c")
        assert result is not None

    def test_strict_missing_bracket(self):
        assert _parse_list("a, b, c", strict=True) is None

    def test_strict_valid(self):
        assert _parse_list('["a", "b", "c"]', strict=True) == ["a", "b", "c"]

    def test_round_trip(self):
        from parallelbench.dataset.task_utils import list_to_str

        original = ["hello", "world", "foo"]
        serialized = list_to_str(original)
        parsed = _parse_list(serialized)
        assert parsed == original

        parsed_strict = _parse_list(serialized, strict=True)
        assert parsed_strict == original


class TestListMatchScore:
    def test_exact_match(self):
        result = list_match_score('["a", "b", "c"]', '["a", "b", "c"]')
        assert result["score"] == 1.0
        assert result["score_strict"] == 1.0

    def test_mismatch(self):
        result = list_match_score('["a", "c", "b"]', '["a", "b", "c"]')
        assert result["score"] == 0.0

    def test_returns_dict(self):
        result = list_match_score('["a"]', '["a"]')
        assert isinstance(result, dict)
        assert "score" in result
        assert "score_strict" in result


class TestDominoScore:
    def test_valid_domino(self):
        gt = {"length": 3, "start": 12}
        result = domino_score('["12", "23", "34"]', gt)
        assert result["score"] == 1.0

    def test_invalid_chain(self):
        gt = {"length": 3, "start": 12}
        result = domino_score('["12", "23", "45"]', gt)
        assert result["score"] == 0.0

    def test_wrong_length(self):
        gt = {"length": 3, "start": 12}
        result = domino_score('["12", "23"]', gt)
        assert result["score"] == 0.0

    def test_returns_dict(self):
        gt = {"length": 2, "start": 12}
        result = domino_score('["12", "23"]', gt)
        assert isinstance(result, dict)
        assert "score" in result and "score_strict" in result


class TestLatinSquareScore:
    def test_valid_3x3(self):
        gt = {"symbols": ["A", "B", "C"]}
        prediction = "A, B, C\nB, C, A\nC, A, B"
        assert latin_square_score(prediction, gt)["score"] == 1.0

    def test_invalid_row(self):
        gt = {"symbols": ["A", "B", "C"]}
        prediction = "A, B, C\nA, B, C\nC, A, B"
        assert latin_square_score(prediction, gt)["score"] == 0.0

    def test_returns_dict(self):
        gt = {"symbols": ["A", "B", "C"]}
        prediction = "A, B, C\nB, C, A\nC, A, B"
        result = latin_square_score(prediction, gt)
        assert isinstance(result, dict)


class TestMathOpScore:
    def test_correct_answer(self):
        result = math_op_score("42", {"result": "42"})
        assert result["score"] == 1.0

    def test_wrong_answer(self):
        result = math_op_score("99", {"result": "42"})
        assert result["score"] == 0.0

    def test_returns_dict(self):
        result = math_op_score("42", {"result": "42"})
        assert isinstance(result, dict)
        assert "score" in result and "score_strict" in result


class TestSudokuScore:
    def test_correct(self):
        gt = "1234\n3412\n2143\n4321"
        result = sudoku_score("1234\n3412\n2143\n4321", gt)
        assert result["score"] == 1.0

    def test_incorrect(self):
        gt = "1234\n3412\n2143\n4321"
        result = sudoku_score("1234\n3412\n2143\n4322", gt)
        assert result["score"] == 0.0


class TestJsonSyntaxScore:
    def test_valid_json(self):
        assert json_syntax_score('{"key": "value"}')["score"] == 1.0

    def test_invalid_json(self):
        assert json_syntax_score("not json")["score"] == 0.0

    def test_empty_prediction(self):
        assert json_syntax_score("")["score"] == 0.0


class TestTextToRegexScore:
    def test_matching_regex(self):
        gt = {
            "positive_examples": ["abc", "abbc"],
            "negative_examples": ["ac", "abcd"],
        }
        result = text_to_regex_score("ab+c", gt)
        assert result["score"] == 1.0

    def test_non_matching_regex(self):
        gt = {
            "positive_examples": ["abc"],
            "negative_examples": ["def"],
        }
        result = text_to_regex_score("[invalid", gt)
        assert result["score"] == 0.0


class TestDictReturningMetrics:
    """Verify metrics that already return dict continue to work."""

    def test_sentence_to_words_score_returns_dict(self):
        try:
            result = sentence_to_words_score(
                "hello world", {"words": ["hello", "world"]}
            )
            assert isinstance(result, dict)
            assert "score" in result
        except ModuleNotFoundError:
            pytest.skip("language_tool_python not installed")

    def test_startwith_score_returns_dict(self):
        try:
            result = startwith_score("Hello world", {"startwith": "Hello"})
            assert isinstance(result, dict)
            assert "score" in result
        except ModuleNotFoundError:
            pytest.skip("language_tool_python not installed")

    def test_regex_match_score_returns_dict(self):
        result = regex_match_score("abc", {"pattern": "abc"})
        assert isinstance(result, dict)
        assert "score" in result


class TestAllMetricsReturnDict:
    """Parametrized test to verify all metric functions return dict[str, float]."""

    @pytest.mark.parametrize(
        "metric_name",
        [
            "list_match_score",
            "list_shuffle_score",
            "domino_score",
            "latin_square_score",
            "math_op_score",
            "text_to_regex_score",
            "json_syntax_score",
            "sudoku_score",
            "regex_match_score",
        ],
    )
    def test_returns_dict(self, metric_name):
        """All metric functions in the map must return dict[str, float]."""
        test_data = {
            "list_match_score": ('["a"]', '["a"]'),
            "list_shuffle_score": ('["b", "a"]', {"input": ["a", "b"]}),
            "domino_score": ('["12", "23"]', {"length": 2, "start": 12}),
            "latin_square_score": ("A, B\nB, A", {"symbols": ["A", "B"]}),
            "math_op_score": ("42", {"result": "42"}),
            "text_to_regex_score": (
                "abc",
                {"positive_examples": ["abc"], "negative_examples": []},
            ),
            "json_syntax_score": ("{}", None),
            "sudoku_score": ("1234\n3412\n2143\n4321", "1234\n3412\n2143\n4321"),
            "regex_match_score": ("abc", {"pattern": "abc"}),
        }

        func = parallel_bench_metric_func_map[metric_name]
        if isinstance(func, type) and issubclass(func, Metric):
            func = func()

        pred, ref = test_data[metric_name]
        result = func(pred, ref)
        assert isinstance(result, dict), (
            f"{metric_name} should return dict, got {type(result)}"
        )
        assert "score" in result, f"{metric_name} result must contain 'score' key"
        for k, v in result.items():
            assert isinstance(v, float), (
                f"{metric_name}[{k}] should be float, got {type(v)}"
            )


class TestMetricFuncMap:
    def test_all_entries_callable(self):
        for name, func in parallel_bench_metric_func_map.items():
            assert callable(func) or (isinstance(func, type)), f"{name} is not callable"
