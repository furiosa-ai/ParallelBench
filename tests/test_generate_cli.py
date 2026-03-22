"""Tests for parallelbench generate CLI subcommand."""

import sys
from unittest import mock

import pytest

from parallelbench.cli.data import generate, main


class TestGenerateArgumentValidation:
    """Test CLI argument validation."""

    def test_push_without_repo_id_raises(self):
        with mock.patch.object(sys, "argv", ["prog", "--push"]):
            with pytest.raises(SystemExit):
                main()

    def test_no_output_or_push_or_dry_run_raises(self):
        with mock.patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                main()

    def test_dry_run_accepted(self):
        """--dry_run alone is a valid invocation."""
        with mock.patch("parallelbench.cli.data.generate", return_value={}) as mock_gen:
            with mock.patch.object(sys, "argv", ["prog", "--dry_run"]):
                main()
            mock_gen.assert_called_once_with(
                output_dir=None,
                push=False,
                repo_id=None,
                private=False,
                dry_run=True,
            )


class TestGenerateFunction:
    """Test the generate() function logic."""

    def test_dry_run_returns_data_without_saving(self, tmp_path):
        result = generate(dry_run=True)
        assert len(result) > 0
        for task_name, rows in result.items():
            assert isinstance(rows, list)
            assert len(rows) > 0

    def test_output_dir_saves_jsonl_files(self, tmp_path):
        result = generate(
            output_dir=str(tmp_path),
        )
        assert len(result) > 0
        for task_name in result:
            jsonl_path = tmp_path / "test" / f"{task_name}.jsonl"
            assert jsonl_path.exists(), f"Expected {jsonl_path} to exist"
            assert jsonl_path.stat().st_size > 0

    def test_generate_returns_data(self):
        result = generate(dry_run=True)
        assert len(result) > 0

    def test_push_calls_hub(self):
        """Verify push triggers push_to_hub with correct args."""
        with (
            mock.patch(
                "parallelbench.datasets.task.create_parallelbench_task"
            ) as mock_create,
            mock.patch("datasets.DatasetDict") as mock_dd,
        ):
            mock_create.return_value = [
                {"input": {"text": "hi"}, "answer": "hello", "metadata": {}}
            ]
            mock_dd_instance = mock.MagicMock()
            mock_dd.return_value = mock_dd_instance

            generate(
                push=True,
                repo_id="test-org/test-repo",
                private=True,
            )

            mock_dd_instance.push_to_hub.assert_called()
            call_kwargs = mock_dd_instance.push_to_hub.call_args
            assert call_kwargs[0][0] == "test-org/test-repo"
            assert call_kwargs[1]["private"] is True


class TestCliRouting:
    """Test that cli routes subcommands correctly."""

    def test_data_routes_to_data_main(self):
        with mock.patch("parallelbench.cli.data.main") as mock_data_main:
            with mock.patch.object(sys, "argv", ["pb", "data", "--dry_run"]):
                from parallelbench.cli import main as cli_main

                cli_main()
            mock_data_main.assert_called_once()

    def test_unknown_command_exits(self):
        with mock.patch.object(sys, "argv", ["pb", "unknown"]):
            from parallelbench.cli import main as cli_main

            with pytest.raises(SystemExit):
                cli_main()

    def test_no_command_exits(self):
        with mock.patch.object(sys, "argv", ["pb"]):
            from parallelbench.cli import main as cli_main

            with pytest.raises(SystemExit):
                cli_main()
