"""Tests for parallelbench.datasets.task_generators registry and generation."""

import random

import pytest

from parallelbench.datasets.task_generators import TASK_GENERATORS
from parallelbench.datasets.task import generate_parallelbench_task_random
from parallelbench.datasets.task_utils import _shuffle


class TestTaskGeneratorRegistry:
    EXPECTED_TYPES = [
        "sort",
        "shuffle",
        "copy",
        "reverse",
        "repeat",
        "insert",
        "remove",
        "replace",
        "domino",
        "math_op",
        "latin_square",
        "rec_cumsum",
        "summary",
        "paraphrase",
    ]

    def test_all_types_registered(self):
        for task_type in self.EXPECTED_TYPES:
            assert task_type in TASK_GENERATORS, (
                f"Task type '{task_type}' not registered"
            )

    def test_all_entries_callable(self):
        for task_type, func in TASK_GENERATORS.items():
            assert callable(func), f"Generator for '{task_type}' is not callable"


class TestSortGenerator:
    def test_deterministic_with_seed(self):
        config = {
            "type": "sort",
            "words": ["apple", "banana", "cherry", "date", "elderberry"],
            "num_samples": 3,
            "min_length": 3,
            "max_length": 4,
        }

        rng1 = random.Random(42)
        result1 = list(generate_parallelbench_task_random(rng1, config))

        rng2 = random.Random(42)
        result2 = list(generate_parallelbench_task_random(rng2, config))

        assert result1 == result2

    def test_sample_structure(self):
        config = {
            "type": "sort",
            "words": ["a", "b", "c", "d"],
            "num_samples": 1,
            "min_length": 3,
            "max_length": 3,
        }
        rng = random.Random(0)
        samples = list(generate_parallelbench_task_random(rng, config))
        assert len(samples) == 1

        sample = samples[0]
        assert "input" in sample
        assert "answer" in sample
        assert "metadata" in sample
        assert "length" in sample["metadata"]


class TestUnknownTaskType:
    def test_raises_on_unknown(self):
        config = {"type": "nonexistent", "num_samples": 1}
        rng = random.Random(0)
        with pytest.raises(ValueError, match="Unknown task type"):
            list(generate_parallelbench_task_random(rng, config))


class TestShuffleEdgeCases:
    def test_empty_list(self):
        rng = random.Random(42)
        assert _shuffle(rng, []) == []

    def test_single_element(self):
        rng = random.Random(42)
        assert _shuffle(rng, [1]) == [1]

    def test_all_identical_elements(self):
        rng = random.Random(42)
        assert _shuffle(rng, [1, 1, 1]) == [1, 1, 1]

    def test_returns_copy(self):
        rng = random.Random(42)
        original = [1]
        result = _shuffle(rng, original)
        assert result is not original

    def test_normal_shuffle_differs(self):
        rng = random.Random(42)
        original = [1, 2, 3, 4, 5]
        result = _shuffle(rng, original)
        assert sorted(result) == sorted(original)
        assert result != original


class TestCLI:
    def test_help_flag(self):
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "-m", "parallelbench.datasets.task", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "task" in result.stdout.lower()
