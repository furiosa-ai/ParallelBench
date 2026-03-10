"""Tests for pb browse CLI subcommand."""

import sys
from unittest import mock

import pytest

from parallelbench.cli.browse import (
    _load_all_task_configs,
    _get_available_task_names,
    _format_value,
    main,
)


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


class TestFormatValue:
    """Test value formatting for display."""

    def test_format_string(self):
        result = _format_value("hello")
        assert "hello" in result

    def test_format_list(self):
        result = _format_value(["a", "b"])
        assert '"a"' in result
        assert '"b"' in result

    def test_format_dict(self):
        result = _format_value({"key": "value"})
        assert "key" in result
        assert "value" in result


class TestBrowseMain:
    """Test CLI entry point."""

    def test_no_args_lists_tasks(self, capsys):
        with mock.patch.object(sys, "argv", ["prog"]):
            main()
        output = capsys.readouterr().out
        assert "Waiting Line" in output
        assert "Text Writing" in output
        assert "Puzzles" in output
        assert "copy" in output

    def test_invalid_task_shows_error(self, capsys):
        with mock.patch.object(sys, "argv", ["prog", "nonexistent/task"]):
            main()
        output = capsys.readouterr().out
        assert "Error" in output
        assert "Unknown task" in output
        assert "waiting_line/copy" in output  # Shows available tasks

    def test_help_flag(self):
        with mock.patch.object(sys, "argv", ["prog", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
