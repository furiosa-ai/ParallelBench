"""Tests for the export module (pb export)."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from parallelbench.export.mapping import (
    MODEL_ID_MAP,
    STRATEGY_ID_MAP,
    TASK_ID_MAP,
    get_model_id,
    get_strategy_id,
    get_task_id,
)
from parallelbench.export.exporter import (
    _KNOWN_AGGREGATE_TASKS,
    _filter_leaf_tasks,
    export_all,
    generate_figures_csv,
    generate_leaderboard_json,
)
from parallelbench.models.unmasking_registry import get_all_methods


# ---------------------------------------------------------------------------
# Mapping completeness tests
# ---------------------------------------------------------------------------


class TestMappingCompleteness:
    """Verify mapping tables cover all known methods and tasks."""

    def test_strategy_map_covers_unmasking_registry(self):
        """All methods in UNMASKING_REGISTRY must have a STRATEGY_ID_MAP entry."""
        registry_methods = get_all_methods()
        mapped_methods = set(STRATEGY_ID_MAP.keys())
        missing = registry_methods - mapped_methods
        assert not missing, f"STRATEGY_ID_MAP is missing methods: {missing}"

    def test_task_map_has_17_leaf_tasks(self):
        """TASK_ID_MAP should have at least 17 entries (the canonical leaf tasks)."""
        # 17 canonical + legacy aliases
        assert len(TASK_ID_MAP) >= 17

    def test_task_map_covers_all_categories(self):
        """TASK_ID_MAP should cover waiting_line, text_writing, and puzzles."""
        task_ids = set(TASK_ID_MAP.values())
        assert any(t.startswith("waitingline-") for t in task_ids)
        assert any(t.startswith("textwriting-") for t in task_ids)
        assert any(t.startswith("puzzles-") for t in task_ids)

    def test_model_map_has_entries(self):
        """MODEL_ID_MAP should have at least the known models."""
        assert len(MODEL_ID_MAP) >= 4  # at least the 4 local models


# ---------------------------------------------------------------------------
# Mapping helper tests
# ---------------------------------------------------------------------------


class TestMappingHelpers:
    """Test mapping lookup functions."""

    def test_get_model_id_known(self):
        assert get_model_id("GSAI-ML/LLaDA-1.5") == "llada15"

    def test_get_model_id_unknown(self):
        assert get_model_id("Unknown/Model") is None

    def test_get_strategy_id_known(self):
        assert get_strategy_id("confidence_topk") == "confidence-topk"

    def test_get_strategy_id_left_to_right(self):
        assert get_strategy_id("left_to_right") == "l2r"

    def test_get_strategy_id_unknown(self):
        assert get_strategy_id("nonexistent_method") is None

    def test_get_task_id_known(self):
        assert get_task_id("parallelbench_waiting_line_copy") == "waitingline-copy"

    def test_get_task_id_puzzles(self):
        assert get_task_id("parallelbench_puzzles_sudoku_n4_12") == "puzzles-sudoku"
        assert (
            get_task_id("parallelbench_puzzles_latin_square_n5")
            == "puzzles-latin_square"
        )

    def test_get_task_id_legacy(self):
        """Legacy task names should also map correctly."""
        assert get_task_id("parallelbench_puzzles_sudoku_n4") == "puzzles-sudoku"
        assert (
            get_task_id("parallelbench_text_writing_words_to_sentence_easy")
            == "textwriting-w2s_easy"
        )

    def test_get_task_id_unknown(self):
        assert get_task_id("parallelbench_unknown_task") is None

    def test_get_task_id_aggregate_not_mapped(self):
        """Aggregate tasks should not be in the task map."""
        assert get_task_id("parallelbench_all") is None
        assert get_task_id("parallelbench_puzzles") is None


# ---------------------------------------------------------------------------
# Leaf task filtering tests
# ---------------------------------------------------------------------------


class TestFilterLeafTasks:
    """Test aggregate row filtering."""

    def test_filters_known_aggregates(self):
        rows = [
            {"task": "parallelbench_all"},
            {"task": "parallelbench_waiting_line"},
            {"task": "parallelbench_waiting_line_copy"},
            {"task": "parallelbench_waiting_line_reverse"},
        ]
        result = _filter_leaf_tasks(rows)
        tasks = {r["task"] for r in result}
        assert "parallelbench_all" not in tasks
        assert "parallelbench_waiting_line" not in tasks
        assert "parallelbench_waiting_line_copy" in tasks
        assert "parallelbench_waiting_line_reverse" in tasks

    def test_prefix_based_filtering(self):
        """Tasks that are prefixes of other tasks should be filtered."""
        rows = [
            {"task": "parallelbench_puzzles"},
            {"task": "parallelbench_puzzles_sudoku_n4_12"},
        ]
        result = _filter_leaf_tasks(rows)
        assert len(result) == 1
        assert result[0]["task"] == "parallelbench_puzzles_sudoku_n4_12"

    def test_all_known_aggregates_filtered(self):
        for agg in _KNOWN_AGGREGATE_TASKS:
            rows = [{"task": agg}, {"task": "parallelbench_waiting_line_copy"}]
            result = _filter_leaf_tasks(rows)
            tasks = {r["task"] for r in result}
            assert agg not in tasks, f"Known aggregate '{agg}' was not filtered"


# ---------------------------------------------------------------------------
# Export format tests (with synthetic data)
# ---------------------------------------------------------------------------


def _make_rows(
    model: str = "GSAI-ML/LLaDA-1.5",
    unmasking: str = "confidence_topk",
    tasks: list[str] | None = None,
    k_values: list[int] | None = None,
) -> list[dict]:
    """Create synthetic result rows for testing."""
    if tasks is None:
        tasks = [
            "parallelbench_waiting_line_copy",
            "parallelbench_waiting_line_reverse",
        ]
    if k_values is None:
        k_values = [1, 2, 4]

    rows = []
    for task in tasks:
        for k in k_values:
            score = max(0, 100 - k * 10)  # score decreases with k
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "unmasking": unmasking,
                    "score": score,
                    "nfe": 128 / k,
                    "tokens_per_step": float(k),
                    "k": float(k),
                    "max_tokens": 128,
                    "steps": 128 // k,
                }
            )
    return rows


class TestLeaderboardJson:
    """Test leaderboard JSON generation."""

    def test_generates_correct_structure(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = generate_leaderboard_json(rows, output_dir)

            assert "llada15" in summary
            output_file = output_dir / "leaderboard" / "llada15.json"
            assert output_file.exists()

            data = json.loads(output_file.read_text())
            assert data["thresholds"] == [80, 75, 70]
            assert "results" in data
            assert "confidence-topk" in data["results"]

    def test_threshold_keys_are_strings(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_leaderboard_json(rows, output_dir)

            data = json.loads((output_dir / "leaderboard" / "llada15.json").read_text())
            strategy_result = data["results"]["confidence-topk"]
            assert all(isinstance(k, str) for k in strategy_result.keys())
            assert set(strategy_result.keys()) == {"80", "75", "70"}

    def test_scores_are_floats(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_leaderboard_json(rows, output_dir)

            data = json.loads((output_dir / "leaderboard" / "llada15.json").read_text())
            for strategy_result in data["results"].values():
                for value in strategy_result.values():
                    assert isinstance(value, (int, float))

    def test_unmapped_model_skipped(self):
        rows = _make_rows(model="Unknown/Model")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = generate_leaderboard_json(rows, output_dir)
            assert len(summary) == 0

    def test_dry_run_writes_no_files(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = generate_leaderboard_json(rows, output_dir, dry_run=True)
            assert "llada15" in summary
            assert not (output_dir / "leaderboard").exists()


class TestFiguresCsv:
    """Test figures CSV generation."""

    def test_generates_csv_with_correct_columns(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_figures_csv(rows, output_dir)

            csv_file = output_dir / "figures" / "llada15" / "confidence-topk.csv"
            assert csv_file.exists()

            with open(csv_file) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ["task", "tps", "accuracy"]

    def test_csv_contains_avg_rows(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_figures_csv(rows, output_dir)

            csv_file = output_dir / "figures" / "llada15" / "confidence-topk.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                csv_rows = list(reader)

            avg_rows = [r for r in csv_rows if r[0] == "avg"]
            assert len(avg_rows) == 3  # 3 k values
            # avg rows should be at the end
            non_avg = [r for r in csv_rows if r[0] != "avg"]
            all_non_avg_first = csv_rows[: len(non_avg)]
            assert all(r[0] != "avg" for r in all_non_avg_first)

    def test_csv_task_names_are_hyphenated(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_figures_csv(rows, output_dir)

            csv_file = output_dir / "figures" / "llada15" / "confidence-topk.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    task = row[0]
                    if task != "avg":
                        assert "parallelbench" not in task
                        assert "-" in task

    def test_csv_sorted_by_task_then_tps(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_figures_csv(rows, output_dir)

            csv_file = output_dir / "figures" / "llada15" / "confidence-topk.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)
                csv_rows = list(reader)

            non_avg = [(r[0], float(r[1])) for r in csv_rows if r[0] != "avg"]
            assert non_avg == sorted(non_avg)

    def test_dry_run_writes_no_files(self):
        rows = _make_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = generate_figures_csv(rows, output_dir, dry_run=True)
            assert len(summary) > 0
            assert not (output_dir / "figures").exists()

    def test_unmapped_model_skipped(self):
        rows = _make_rows(model="Unknown/Model")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = generate_figures_csv(rows, output_dir)
            assert len(summary) == 0


class TestExportAll:
    """Test the full export_all orchestration with real results."""

    @pytest.fixture
    def results_dir(self):
        """Return the real results directory if it exists."""
        path = Path("results")
        if not path.exists() or not list(path.rglob("results_*.json")):
            pytest.skip("No results directory available")
        return path

    def test_export_all_produces_output(self, results_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = export_all(results_dir, output_dir)

            assert "leaderboard" in summary
            assert "figures" in summary
            assert len(summary["leaderboard"]) > 0
            assert len(summary["figures"]) > 0

            # Check files exist
            leaderboard_dir = output_dir / "leaderboard"
            assert leaderboard_dir.exists()
            json_files = list(leaderboard_dir.glob("*.json"))
            assert len(json_files) > 0

            figures_dir = output_dir / "figures"
            assert figures_dir.exists()
            csv_files = list(figures_dir.rglob("*.csv"))
            assert len(csv_files) > 0

    def test_dry_run_produces_no_files(self, results_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = export_all(results_dir, output_dir, dry_run=True)

            assert len(summary["leaderboard"]) > 0
            assert not (output_dir / "leaderboard").exists()
            assert not (output_dir / "figures").exists()
