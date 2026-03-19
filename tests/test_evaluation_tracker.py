"""Tests for timestamp-based run directory naming and .latest symlink removal."""

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path


TIMESTAMP_RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{4}$")


# ---------------------------------------------------------------------------
# Step 1: eval.py — _extract_run_name()
# ---------------------------------------------------------------------------


class TestExtractRunName:
    """Tests for _extract_run_name() in parallelbench/cli/eval.py."""

    def _call(self):
        from parallelbench.cli.eval import _extract_run_name

        return _extract_run_name()

    def test_generate_run_name_default_timestamp(self):
        """Default (no --run_name) generates YYYYMMDD_HHMMSS_XXXX."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval", "--model", "foo"]
            run_name = self._call()
            assert TIMESTAMP_RUN_NAME_RE.match(run_name), (
                f"Expected YYYYMMDD_HHMMSS_XXXX format, got: {run_name}"
            )
        finally:
            sys.argv = original_argv

    def test_generate_run_name_unique(self):
        """Two rapid calls produce different run names (random suffix differs)."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval"]
            name1 = self._call()
            sys.argv = ["pb", "eval"]
            name2 = self._call()
            assert name1 != name2, f"Expected different names, got: {name1} and {name2}"
        finally:
            sys.argv = original_argv

    def test_extract_run_name_custom_value(self):
        """--run_name passes through the custom value."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval", "--run_name", "my_experiment"]
            run_name = self._call()
            assert run_name == "my_experiment"
        finally:
            sys.argv = original_argv

    def test_extract_run_name_custom_value_equals_syntax(self):
        """--run_name=value syntax also works."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval", "--run_name=custom_run"]
            run_name = self._call()
            assert run_name == "custom_run"
        finally:
            sys.argv = original_argv

    def test_extract_run_name_from_env_var(self, monkeypatch):
        """PB_RUN_NAME env var is used when --run_name is not provided."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval"]
            monkeypatch.setenv("PB_RUN_NAME", "env_run_name")
            run_name = self._call()
            assert run_name == "env_run_name"
        finally:
            sys.argv = original_argv

    def test_extract_run_name_flag_overrides_env_var(self, monkeypatch):
        """--run_name flag takes priority over PB_RUN_NAME env var."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["pb", "eval", "--run_name", "flag_value"]
            monkeypatch.setenv("PB_RUN_NAME", "env_value")
            run_name = self._call()
            assert run_name == "flag_value"
        finally:
            sys.argv = original_argv


# ---------------------------------------------------------------------------
# Step 2: evaluation_tracker.py — _resolve_output_dir()
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    """Tests for ParallelBenchEvaluationTracker._resolve_output_dir()."""

    def test_resolve_output_dir_creates_timestamp_dir(self, tmp_path):
        """Output dir uses the run_name (timestamp format)."""
        from parallelbench.lm_eval_wrappers.evaluation_tracker import (
            ParallelBenchEvaluationTracker,
        )

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        ParallelBenchEvaluationTracker.run_name = run_name

        tracker = ParallelBenchEvaluationTracker.__new__(ParallelBenchEvaluationTracker)
        tracker.output_path = str(tmp_path)

        # Mock the general_config_tracker
        class MockConfig:
            model_name_sanitized = "test_model"

        tracker.general_config_tracker = MockConfig()

        result = tracker._resolve_output_dir(None)
        assert result.name == run_name
        assert result.exists()

    def test_resolve_output_dir_no_latest_symlink(self, tmp_path):
        """No .latest symlink is created after resolving output dir."""
        from parallelbench.lm_eval_wrappers.evaluation_tracker import (
            ParallelBenchEvaluationTracker,
        )

        run_name = "20260319_143052_a3f2"
        ParallelBenchEvaluationTracker.run_name = run_name

        tracker = ParallelBenchEvaluationTracker.__new__(ParallelBenchEvaluationTracker)
        tracker.output_path = str(tmp_path)

        class MockConfig:
            model_name_sanitized = "test_model"

        tracker.general_config_tracker = MockConfig()

        tracker._resolve_output_dir(None)

        # Check that no .latest symlink exists anywhere in the output tree
        latest_links = list(tmp_path.rglob(".latest"))
        assert len(latest_links) == 0, (
            f"Expected no .latest symlinks, found: {latest_links}"
        )


# ---------------------------------------------------------------------------
# Step 3: analyze.py — _find_latest_run_dirs()
# ---------------------------------------------------------------------------


def _create_results_file(run_dir: Path, task_name: str = "test_task") -> Path:
    """Helper: create a minimal results_*.json file in a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / f"results_{task_name}.json"
    results_file.write_text(
        json.dumps(
            {
                "results": {task_name: {"score,none": 85.0}},
                "config": {"model": "test"},
            }
        ),
        encoding="utf-8",
    )
    return results_file


class TestFindLatestResultFiles:
    """Tests for _find_latest_result_files() in parallelbench/cli/analyze.py."""

    def test_finds_latest_by_sorting(self, tmp_path):
        """Lexicographically later timestamp dir is selected as latest."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "20260101_120000_aaaa")
        _create_results_file(repr_param / "20260102_120000_bbbb")

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 1
        assert latest[0].parent.name == "20260102_120000_bbbb"

    def test_mixed_uuid_and_timestamp_dirs(self, tmp_path):
        """Timestamp dir is preferred over legacy UUID dir."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "a3f2b1c0")  # legacy UUID
        _create_results_file(repr_param / "20260319_143052_a3f2")  # timestamp

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 1
        assert latest[0].parent.name == "20260319_143052_a3f2"

    def test_legacy_uuid_only_dirs(self, tmp_path):
        """When only UUID dirs exist, falls back to lex sort of all dirs."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "a3f2b1c0")
        _create_results_file(repr_param / "b4e2c1d0")

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 1
        assert latest[0].parent.name == "b4e2c1d0"

    def test_separate_runs_per_task_all_found(self, tmp_path):
        """Different tasks in different run dirs are all selected."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "20260319_140000_aaaa", "puzzles")
        _create_results_file(repr_param / "20260319_150000_bbbb", "waiting_line")
        _create_results_file(repr_param / "20260319_160000_cccc", "text_writing")

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 3
        names = {f.name for f in latest}
        assert names == {
            "results_puzzles.json",
            "results_waiting_line.json",
            "results_text_writing.json",
        }

    def test_separate_runs_same_task_picks_latest(self, tmp_path):
        """Same task in multiple runs: only the latest is selected."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "20260319_140000_aaaa", "puzzles")
        _create_results_file(repr_param / "20260319_150000_bbbb", "puzzles")

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 1
        assert latest[0].parent.name == "20260319_150000_bbbb"

    def test_rerun_task_picks_newer(self, tmp_path):
        """When a task is rerun, the newer result file is selected."""
        from parallelbench.cli.analyze import _find_latest_result_files

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        # First run: all tasks
        _create_results_file(repr_param / "20260319_140000_aaaa", "puzzles")
        _create_results_file(repr_param / "20260319_140000_aaaa", "waiting_line")
        # Second run: only puzzles rerun
        _create_results_file(repr_param / "20260319_150000_bbbb", "puzzles")

        latest = _find_latest_result_files(tmp_path)
        assert len(latest) == 2
        latest_by_task = {f.name: f.parent.name for f in latest}
        assert latest_by_task["results_puzzles.json"] == "20260319_150000_bbbb"
        assert latest_by_task["results_waiting_line.json"] == "20260319_140000_aaaa"


# ---------------------------------------------------------------------------
# Step 4: analyze.py — _compute_average_rows()
# ---------------------------------------------------------------------------


class TestComputeAverageRows:
    """Tests for _compute_average_rows() in parallelbench/cli/analyze.py."""

    def test_no_average_for_single_task(self):
        """No average rows when only one task exists."""
        from parallelbench.cli.analyze import _compute_average_rows

        rows = [
            {"task": "puzzles", "alg_threshold": "0.5", "score": 85.0, "nfe": 10},
            {"task": "puzzles", "alg_threshold": "0.6", "score": 80.0, "nfe": 10},
        ]
        assert _compute_average_rows(rows, "threshold") == []

    def test_average_per_hyperparameter(self):
        """Average is computed per hyperparameter value across tasks."""
        from parallelbench.cli.analyze import _compute_average_rows

        rows = [
            {"task": "puzzles", "alg_threshold": "0.5", "score": 80.0, "nfe": 10},
            {"task": "waiting_line", "alg_threshold": "0.5", "score": 90.0, "nfe": 10},
            {"task": "puzzles", "alg_threshold": "0.6", "score": 70.0, "nfe": 10},
            {"task": "waiting_line", "alg_threshold": "0.6", "score": 80.0, "nfe": 10},
        ]
        avg_rows = _compute_average_rows(rows, "threshold")
        assert len(avg_rows) == 2

        avg_by_threshold = {r["alg_threshold"]: r for r in avg_rows}
        assert avg_by_threshold["0.5"]["score"] == 85.0
        assert avg_by_threshold["0.6"]["score"] == 75.0
        assert all(r["task"] == "Average" for r in avg_rows)

    def test_average_topk_method(self):
        """Average works for topk method type grouped by k."""
        from parallelbench.cli.analyze import _compute_average_rows

        rows = [
            {"task": "puzzles", "k": "4", "score": 80.0, "nfe": 10},
            {"task": "waiting_line", "k": "4", "score": 90.0, "nfe": 10},
        ]
        avg_rows = _compute_average_rows(rows, "topk")
        assert len(avg_rows) == 1
        assert avg_rows[0]["k"] == "4"
        assert avg_rows[0]["score"] == 85.0

    def test_no_average_for_unknown_method(self):
        """Unknown method type returns no averages."""
        from parallelbench.cli.analyze import _compute_average_rows

        rows = [
            {"task": "puzzles", "score": 80.0, "nfe": 10},
            {"task": "waiting_line", "score": 90.0, "nfe": 10},
        ]
        assert _compute_average_rows(rows, "unknown") == []

    def test_excludes_group_level_tasks(self):
        """Group-level aggregate rows are excluded from average."""
        from parallelbench.cli.analyze import _compute_average_rows

        rows = [
            # Group-level aggregate — should be excluded (prefix of subtasks)
            {"task": "puzzles", "alg_threshold": "0.5", "score": 30.0, "nfe": 10},
            # Leaf tasks
            {
                "task": "puzzles_sudoku",
                "alg_threshold": "0.5",
                "score": 80.0,
                "nfe": 10,
            },
            {"task": "puzzles_latin", "alg_threshold": "0.5", "score": 90.0, "nfe": 10},
        ]
        avg_rows = _compute_average_rows(rows, "threshold")
        assert len(avg_rows) == 1
        assert avg_rows[0]["score"] == 85.0  # (80 + 90) / 2, not (30 + 80 + 90) / 3
