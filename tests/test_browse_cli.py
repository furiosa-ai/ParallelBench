"""Tests for pb browse CLI subcommand."""

import sys
from io import StringIO
from unittest import mock

import pytest
from rich.console import Console

from parallelbench.cli.browse import (
    _load_all_task_configs,
    _get_available_task_names,
    main,
)


def _capture_rich_output(func, *args, **kwargs):
    """Run a function while capturing Rich console output."""
    string_io = StringIO()
    test_console = Console(file=string_io, force_terminal=False, highlight=False)
    with mock.patch("parallelbench.cli.browse.console", test_console):
        func(*args, **kwargs)
    return string_io.getvalue()


class TestLoadAllTaskConfigs:
    """Test YAML config loading for task listing."""

    def test_returns_dict(self):
        configs = _load_all_task_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0

    def test_all_tasks_have_category_prefix(self):
        configs = _load_all_task_configs()
        for name in configs:
            assert "/" in name, f"Task name '{name}' missing category prefix"

    def test_configs_have_num_samples(self):
        configs = _load_all_task_configs()
        for name, cfg in configs.items():
            assert "num_samples" in cfg, f"Task '{name}' missing num_samples"

    def test_configs_have_metric(self):
        configs = _load_all_task_configs()
        for name, cfg in configs.items():
            assert "metric" in cfg, f"Task '{name}' missing metric"


class TestGetAvailableTaskNames:
    """Test task name listing."""

    def test_returns_sorted_list(self):
        names = _get_available_task_names()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_excludes_underscore_prefix(self):
        names = _get_available_task_names()
        for name in names:
            assert not name.split("/")[-1].startswith("_")

    def test_contains_known_tasks(self):
        names = _get_available_task_names()
        assert "waiting_line/copy" in names
        assert "puzzles/sudoku_n4" in names
        assert "text_writing/paraphrasing" in names


class TestBrowseMain:
    """Test CLI entry point."""

    def test_no_args_lists_tasks(self):
        with mock.patch.object(sys, "argv", ["prog"]):
            output = _capture_rich_output(main)
        assert "Waiting Line" in output
        assert "Text Writing" in output
        assert "Puzzles" in output
        assert "copy" in output

    def test_invalid_task_shows_error(self):
        with mock.patch.object(sys, "argv", ["prog", "nonexistent/task"]):
            output = _capture_rich_output(main)
        assert "Error" in output
        assert "Unknown task" in output
        assert "waiting_line/copy" in output  # Shows available tasks

    def test_help_flag(self):
        with mock.patch.object(sys, "argv", ["prog", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
