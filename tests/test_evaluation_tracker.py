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


class TestFindLatestRunDirs:
    """Tests for _find_latest_run_dirs() in parallelbench/cli/analyze.py."""

    def test_analyze_finds_latest_by_sorting(self, tmp_path):
        """Lexicographically later timestamp dir is selected as latest."""
        from parallelbench.cli.analyze import _find_latest_run_dirs

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "20260101_120000_aaaa")
        _create_results_file(repr_param / "20260102_120000_bbbb")

        latest = _find_latest_run_dirs(tmp_path)
        assert len(latest) == 1
        assert latest[0].name == "20260102_120000_bbbb"

    def test_analyze_mixed_uuid_and_timestamp_dirs(self, tmp_path):
        """Timestamp dir is preferred over legacy UUID dir.

        UUID 'a3f2b1c0' sorts after '20260319...' in ASCII (a > 2),
        but the regex filter ensures only timestamp dirs are considered.
        """
        from parallelbench.cli.analyze import _find_latest_run_dirs

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "a3f2b1c0")  # legacy UUID
        _create_results_file(repr_param / "20260319_143052_a3f2")  # timestamp

        latest = _find_latest_run_dirs(tmp_path)
        assert len(latest) == 1
        assert latest[0].name == "20260319_143052_a3f2"

    def test_analyze_legacy_uuid_only_dirs(self, tmp_path):
        """When only UUID dirs exist, falls back to lex sort of all dirs."""
        from parallelbench.cli.analyze import _find_latest_run_dirs

        repr_param = tmp_path / "model" / "unmasking" / "k4"
        _create_results_file(repr_param / "a3f2b1c0")
        _create_results_file(repr_param / "b4e2c1d0")

        latest = _find_latest_run_dirs(tmp_path)
        assert len(latest) == 1
        assert latest[0].name == "b4e2c1d0"
