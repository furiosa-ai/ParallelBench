"""Tests for flex mode: on-the-fly data generation with configurable difficulty."""

import pytest

from parallelbench.dataset import ParallelBench
from parallelbench.dataset.task import load_task_flex, _flex_seed


class TestFlexSeed:
    """Seed determinism and isolation tests."""

    def test_same_config_produces_same_seed(self):
        config = {"min_length": 5, "max_length": 10}
        seed1 = _flex_seed("waiting_line/copy", config)
        seed2 = _flex_seed("waiting_line/copy", config)
        assert seed1 == seed2

    def test_different_config_produces_different_seed(self):
        config_a = {"min_length": 3, "max_length": 6}
        config_b = {"min_length": 10, "max_length": 15}
        seed_a = _flex_seed("waiting_line/copy", config_a)
        seed_b = _flex_seed("waiting_line/copy", config_b)
        assert seed_a != seed_b

    def test_different_task_produces_different_seed(self):
        config = {"min_length": 3, "max_length": 6}
        seed_copy = _flex_seed("waiting_line/copy", config)
        seed_reverse = _flex_seed("waiting_line/reverse", config)
        assert seed_copy != seed_reverse

    def test_seed_in_valid_range(self):
        config = {"min_length": 5, "max_length": 10}
        seed = _flex_seed("waiting_line/copy", config)
        assert 0 <= seed < 2**16 - 1


class TestLoadTaskFlex:
    """Tests for load_task_flex function."""

    def test_generates_correct_sample_count(self):
        ds, config = load_task_flex(
            "test",
            "waiting_line/copy",
            flex_config={
                "min_length": 3,
                "max_length": 3,
                "num_samples": 10,
                "samples_per_length": 10,
            },
        )
        assert len(ds) == 10

    def test_respects_length_override(self):
        ds, config = load_task_flex(
            "test",
            "waiting_line/copy",
            flex_config={
                "min_length": 5,
                "max_length": 5,
                "num_samples": 10,
                "samples_per_length": 10,
            },
        )
        for sample in ds:
            words = sample["input"]["context"].split(", ")
            assert len(words) == 5

    def test_deterministic_across_calls(self):
        flex = {
            "min_length": 3,
            "max_length": 3,
            "num_samples": 5,
            "samples_per_length": 5,
        }
        ds1, _ = load_task_flex("test", "waiting_line/copy", flex_config=flex)
        ds2, _ = load_task_flex("test", "waiting_line/copy", flex_config=flex)
        for i in range(len(ds1)):
            assert ds1[i]["input"] == ds2[i]["input"]
            assert ds1[i]["answer"] == ds2[i]["answer"]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task for flex mode"):
            load_task_flex(
                "test", "waiting_line/nonexistent", flex_config={"min_length": 3}
            )

    def test_returns_task_config_with_required_keys(self):
        _, config = load_task_flex(
            "test",
            "waiting_line/copy",
            flex_config={
                "min_length": 3,
                "max_length": 3,
                "num_samples": 5,
                "samples_per_length": 5,
            },
        )
        assert "prompt" in config
        assert "metric" in config
        assert "name" in config


class TestParallelBenchFlexIntegration:
    """Integration tests: ParallelBench with flex_config."""

    def test_flex_config_none_uses_original_path(self):
        """When flex_config is None, the original load_task path is used."""
        pb = ParallelBench(task="waiting_line/copy", split="test")
        assert len(pb) > 0

    def test_flex_config_generates_data(self):
        pb = ParallelBench(
            task="waiting_line/copy",
            split="test",
            flex_config={
                "min_length": 4,
                "max_length": 4,
                "num_samples": 8,
                "samples_per_length": 8,
            },
        )
        assert len(pb) == 8
        sample = pb[0]
        assert "input" in sample
        assert "label" in sample
        assert "metadata" in sample

    def test_flex_config_different_from_default(self):
        """Flex data with custom params should differ from default data."""
        pb_flex = ParallelBench(
            task="waiting_line/copy",
            split="test",
            flex_config={
                "min_length": 10,
                "max_length": 10,
                "num_samples": 5,
                "samples_per_length": 5,
            },
        )
        # Flex with length 10 should have longer lists than default (3-6)
        flex_sample = pb_flex[0]
        words = flex_sample["input"]["messages"][-1]["content"]
        # Length 10 lists should have more commas than length 3-6 lists
        assert words.count(",") >= 9  # 10 items = 9 commas


class TestFlexWithPuzzles:
    """Flex mode for puzzle tasks."""

    def test_latin_square_flex(self):
        pb = ParallelBench(
            task="puzzles/latin_square_n4",
            split="test",
            flex_config={"size": 4, "num_samples": 5, "samples_per_length": 5},
        )
        assert len(pb) == 5
