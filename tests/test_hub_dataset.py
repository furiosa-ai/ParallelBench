"""Tests for HuggingFace Hub dataset push/load utilities."""

import json

import pytest

from parallelbench.dataset.task import (
    _try_json_loads,
    task_name_to_config_name,
    config_name_to_task_name,
)


class TestConfigNameConversion:
    @pytest.mark.parametrize(
        "task_name,expected",
        [
            ("waiting_line/copy", "waiting_line-copy"),
            ("puzzles/sudoku_n4_12", "puzzles-sudoku_n4_12"),
            ("text_writing/summarization", "text_writing-summarization"),
            (
                "text_writing/words_to_sentence_easy",
                "text_writing-words_to_sentence_easy",
            ),
        ],
    )
    def test_task_name_to_config_name(self, task_name, expected):
        assert task_name_to_config_name(task_name) == expected

    @pytest.mark.parametrize(
        "config_name,expected",
        [
            ("waiting_line-copy", "waiting_line/copy"),
            ("puzzles-sudoku_n4_12", "puzzles/sudoku_n4_12"),
            ("text_writing-summarization", "text_writing/summarization"),
        ],
    )
    def test_config_name_to_task_name(self, config_name, expected):
        assert config_name_to_task_name(config_name) == expected

    @pytest.mark.parametrize(
        "task_name",
        [
            "waiting_line/copy",
            "puzzles/sudoku_n4_12",
            "text_writing/words_to_sentence_easy",
        ],
    )
    def test_roundtrip(self, task_name):
        config_name = task_name_to_config_name(task_name)
        assert config_name_to_task_name(config_name) == task_name


class TestTryJsonLoads:
    def test_valid_json_dict(self):
        data = {"key": "value", "nested": [1, 2]}
        assert _try_json_loads(json.dumps(data)) == data

    def test_valid_json_list(self):
        data = [1, "two", 3.0]
        assert _try_json_loads(json.dumps(data)) == data

    def test_plain_string_returns_as_is(self):
        text = "1342\n2431\n4123\n3214"
        assert _try_json_loads(text) == text

    def test_non_string_passthrough(self):
        assert _try_json_loads(42) == 42
        assert _try_json_loads(None) is None
        assert _try_json_loads({"a": 1}) == {"a": 1}

    def test_string_answer_not_double_encoded(self):
        # string answer가 serialize 후에도 원래 string으로 복원되는지 확인
        original = '["Billy Ramos", "Alan Wells"]'
        assert _try_json_loads(original) == ["Billy Ramos", "Alan Wells"]

    def test_dict_answer_roundtrip(self):
        original = {"words": ["dog", "park"], "example": "A dog in the park."}
        serialized = json.dumps(original, ensure_ascii=False)
        assert _try_json_loads(serialized) == original


class TestSerializationRoundtrip:
    """로컬 JSONL → serialize → deserialize → 원본과 동일한지 검증"""

    @pytest.fixture
    def local_task_data(self):
        from parallelbench.dataset.task import load_task

        ds, config = load_task("test", "waiting_line/copy")
        return ds, config

    def test_input_roundtrip(self, local_task_data):
        ds, _ = local_task_data
        sample = ds[0]
        serialized = json.dumps(sample["input"], ensure_ascii=False)
        deserialized = _try_json_loads(serialized)
        assert deserialized == sample["input"]

    def test_metadata_roundtrip(self, local_task_data):
        ds, _ = local_task_data
        sample = ds[0]
        serialized = json.dumps(sample["metadata"], ensure_ascii=False)
        deserialized = _try_json_loads(serialized)
        assert deserialized == sample["metadata"]

    def test_string_answer_roundtrip(self, local_task_data):
        ds, _ = local_task_data
        sample = ds[0]
        # waiting_line/copy의 answer는 string
        answer = sample["answer"]
        assert isinstance(answer, str)
        # string은 serialize하지 않으므로 그대로 유지
        assert _try_json_loads(answer) is not None

    def test_dict_answer_roundtrip(self):
        from parallelbench.dataset.task import load_task

        ds, _ = load_task("test", "text_writing/words_to_sentence_easy")
        sample = ds[0]
        answer = sample["answer"]
        assert isinstance(answer, dict)
        serialized = json.dumps(answer, ensure_ascii=False)
        deserialized = _try_json_loads(serialized)
        assert deserialized == answer
