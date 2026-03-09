"""Tests for parallelbench.datasets module (ParallelBench class and supporting utilities)."""

import pytest

from parallelbench.datasets import (
    ParallelBench,
    get_task_names,
    PARALLEL_BENCH_TASKS,
    PARALLEL_BENCH_MASK_TOKEN,
)


class TestGetTaskNames:
    def test_returns_list(self):
        names = get_task_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_no_hidden_tasks(self):
        for name in get_task_names():
            assert not name.startswith("_")

    def test_parallel_bench_tasks_populated(self):
        assert len(PARALLEL_BENCH_TASKS) > 0


class TestMaskToken:
    def test_mask_token_value(self):
        assert PARALLEL_BENCH_MASK_TOKEN == "[MASK]"

    def test_importable_from_both_paths(self):
        from parallelbench.datasets import PARALLEL_BENCH_MASK_TOKEN as token1
        from parallelbench.datasets.task import PARALLEL_BENCH_MASK_TOKEN as token2

        assert token1 == token2


SIMPLE_TASK = "waiting_line/copy"


class TestParallelBench:
    @pytest.fixture
    def benchmark(self):
        return ParallelBench(SIMPLE_TASK, num_samples=2)

    def test_len(self, benchmark):
        assert len(benchmark) == 2

    def test_getitem_keys(self, benchmark):
        sample = benchmark[0]
        assert set(sample.keys()) == {"input", "label", "index", "metadata"}

    def test_getitem_input_has_messages(self, benchmark):
        sample = benchmark[0]
        assert "messages" in sample["input"]
        assert len(sample["input"]["messages"]) > 0

    def test_compute_metrics_returns_dict(self, benchmark):
        sample = benchmark[0]
        label = sample["label"]

        if isinstance(label, dict):
            prediction = label.get("example", label.get("result", str(label)))
        else:
            prediction = str(label)

        metrics = benchmark.compute_metrics([prediction], [label])
        assert isinstance(metrics, dict)
        assert "score" in metrics

    def test_compute_metrics_with_per_sample(self, benchmark):
        sample = benchmark[0]
        label = sample["label"]

        if isinstance(label, dict):
            prediction = label.get("example", label.get("result", str(label)))
        else:
            prediction = str(label)

        metrics, per_sample = benchmark.compute_metrics(
            [prediction], [label], output_per_sample=True
        )
        assert isinstance(per_sample, list)
        assert len(per_sample) == 1
        assert isinstance(per_sample[0], dict)
