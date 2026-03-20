"""Tests for explicit ICL (In-Context Learning) example system.

Verifies that:
- Static mode uses explicit ICL examples from task config
- ICL examples are shared across all samples in a task
- ICL examples do not overlap with test data
- Flex mode generates ICL examples on-the-fly
- Missing icl_example raises ValueError when icl_example_count > 0
"""

import json
import random

import pytest

from parallelbench.datasets import ParallelBench
from parallelbench.datasets.task import (
    _generate_icl_example_on_the_fly,
    _validate_icl_not_in_data,
    create_parallelbench_task,
    load_task_flex,
)
from parallelbench.datasets.task_utils import load_task_configs


class TestValidateIclNotInData:
    """Tests for _validate_icl_not_in_data helper."""

    def test_no_overlap_passes(self):
        icl = {"input": {"context": "icl_context"}, "answer": "icl_answer"}
        data = [
            {"input": {"context": "test_context_1"}},
            {"input": {"context": "test_context_2"}},
        ]
        # Should not raise
        _validate_icl_not_in_data(icl, data, "test_task")

    def test_overlap_raises(self):
        icl = {"input": {"context": "same_context"}, "answer": "icl_answer"}
        data = [
            {"input": {"context": "different"}},
            {"input": {"context": "same_context"}},
        ]
        with pytest.raises(ValueError, match="overlaps with test sample"):
            _validate_icl_not_in_data(icl, data, "test_task")


class TestGenerateIclExampleOnTheFly:
    """Tests for _generate_icl_example_on_the_fly helper."""

    def test_returns_input_and_answer(self):
        configs = load_task_configs("test/waiting_line")
        task_config = {
            **configs["waiting_line/copy"],
            "num_samples": 3,
            "samples_per_length": 0,
            "icl_example_count": 0,
        }
        rng = random.Random(42)
        from parallelbench.datasets.task import _create_task
        from parallelbench.datasets.task_utils import load_words_from_file

        if "words" in task_config and isinstance(task_config["words"], str):
            task_config["words"] = load_words_from_file(task_config["words"])
        task_config["seed"] = 42
        data = _create_task(rng, task_config)

        icl = _generate_icl_example_on_the_fly(rng, task_config, data)
        assert "input" in icl
        assert "answer" in icl

    def test_not_in_data(self):
        configs = load_task_configs("test/waiting_line")
        task_config = {
            **configs["waiting_line/copy"],
            "num_samples": 5,
            "samples_per_length": 0,
            "icl_example_count": 0,
        }
        rng = random.Random(42)
        from parallelbench.datasets.task import _create_task
        from parallelbench.datasets.task_utils import load_words_from_file

        if "words" in task_config and isinstance(task_config["words"], str):
            task_config["words"] = load_words_from_file(task_config["words"])
        task_config["seed"] = 42
        data = _create_task(rng, task_config)

        icl = _generate_icl_example_on_the_fly(rng, task_config, data)
        icl_key = json.dumps(icl["input"], sort_keys=True)
        for sample in data:
            assert json.dumps(sample["input"], sort_keys=True) != icl_key


class TestStaticIclExample:
    """Tests for static ICL example from task config."""

    def test_explicit_icl_used(self):
        configs = load_task_configs("test/waiting_line")
        task_config = {
            **configs["waiting_line/copy"],
            "num_samples": 3,
            "samples_per_length": 0,
        }
        data = create_parallelbench_task(
            split="test", task=task_config, output_file=None, no_save=True
        )
        assert "icl_examples" in data[0]["input"]
        icl = data[0]["input"]["icl_examples"][0]
        assert icl == task_config["icl_example"]

    def test_all_samples_share_same_icl(self):
        configs = load_task_configs("test/waiting_line")
        task_config = {
            **configs["waiting_line/copy"],
            "num_samples": 5,
            "samples_per_length": 0,
        }
        data = create_parallelbench_task(
            split="test", task=task_config, output_file=None, no_save=True
        )
        first_icl = data[0]["input"]["icl_examples"]
        for sample in data:
            assert sample["input"]["icl_examples"] == first_icl

    def test_missing_icl_example_raises(self):
        configs = load_task_configs("test/waiting_line")
        task_config = {
            **configs["waiting_line/copy"],
            "num_samples": 3,
            "samples_per_length": 0,
        }
        # Remove the explicit icl_example but keep icl_example_count > 0
        task_config.pop("icl_example", None)
        with pytest.raises(ValueError, match="no icl_example or icl_examples defined"):
            create_parallelbench_task(
                split="test", task=task_config, output_file=None, no_save=True
            )

    def test_puzzles_icl_example(self):
        configs = load_task_configs("test/puzzles")
        task_config = {
            **configs["puzzles/sudoku_n4"],
            "num_samples": 3,
            "samples_per_length": 0,
        }
        data = create_parallelbench_task(
            split="test", task=task_config, output_file=None, no_save=True
        )
        icl = data[0]["input"]["icl_examples"][0]
        assert "puzzle" in icl["input"]
        assert isinstance(icl["answer"], str)


class TestFlexIclExample:
    """Tests for on-the-fly ICL in flex mode."""

    def test_flex_generates_icl(self):
        ds, _ = load_task_flex(
            "test",
            "waiting_line/copy",
            {"num_samples": 5, "min_length": 4, "max_length": 4},
        )
        assert "icl_examples" in ds[0]["input"]

    def test_flex_icl_not_in_test_data(self):
        ds, _ = load_task_flex(
            "test",
            "waiting_line/copy",
            {"num_samples": 10, "min_length": 3, "max_length": 3},
        )
        icl = ds[0]["input"]["icl_examples"][0]
        icl_key = json.dumps(icl["input"], sort_keys=True)
        for sample in ds:
            sample_input = {
                k: v for k, v in sample["input"].items() if k != "icl_examples"
            }
            assert json.dumps(sample_input, sort_keys=True) != icl_key

    def test_flex_icl_deterministic(self):
        flex = {"num_samples": 5, "min_length": 3, "max_length": 3}
        ds1, _ = load_task_flex("test", "waiting_line/copy", flex)
        ds2, _ = load_task_flex("test", "waiting_line/copy", flex)
        icl1 = ds1[0]["input"]["icl_examples"][0]
        icl2 = ds2[0]["input"]["icl_examples"][0]
        assert icl1 == icl2


class TestIclInParallelBenchClass:
    """Integration: ICL example flows through ParallelBench.__getitem__."""

    def test_messages_include_icl(self):
        pb = ParallelBench(
            "waiting_line/sort",
            split="test",
            flex_config={"num_samples": 3, "min_length": 3, "max_length": 3},
        )
        sample = pb[0]
        messages = sample["input"]["messages"]
        # 3 messages: icl_user, icl_assistant, actual_user
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_icl_zero_count_no_icl_messages(self):
        """Tasks without ICL should have only 1 message (the actual query)."""
        pb = ParallelBench(
            "waiting_line/copy",
            split="test",
            flex_config={
                "num_samples": 3,
                "min_length": 3,
                "max_length": 3,
                "icl_example_count": 0,
            },
        )
        sample = pb[0]
        messages = sample["input"]["messages"]
        assert len(messages) == 1
