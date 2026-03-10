"""Generate ParallelBench data and optionally push to HuggingFace Hub.

Usage via CLI:
    pb data --output_dir ./output
    pb data --push --repo_id org/name
    pb data --output_dir ./output --push --repo_id org/name
    pb data --dry_run
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

console = Console()

_TASK_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "data"
    / "task_configs"
    / "test"
)


def _serialize_column(value):
    """Serialize dict/list to JSON string for Hub storage."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _load_all_tasks() -> dict[str, dict]:
    """Load all task configs.

    Discovers YAML files under data/task_configs/test/ and returns
    a flat dict mapping task_name -> task_config.
    """
    from parallelbench.datasets.task_utils import load_task_configs

    if not _TASK_CONFIG_DIR.exists():
        return {}

    tasks = {}
    for yaml_file in sorted(_TASK_CONFIG_DIR.glob("*.yaml")):
        loaded = load_task_configs(str(yaml_file))
        tasks.update(loaded)
    return tasks


def _build_hub_dataset(task_name: str, task_config: dict, rows: list[dict]):
    """Convert generated rows into a Hub-ready Dataset with serialized columns."""
    from datasets import Dataset

    prompt = task_config.get("prompt", "")
    metric = task_config.get("metric", "")

    records = []
    for row in rows:
        records.append(
            {
                "input": _serialize_column(row["input"]),
                "answer": _serialize_column(row["answer"]),
                "output_format": row.get("output_format"),
                "metadata": _serialize_column(row["metadata"]),
                "task_name": task_name,
                "prompt": prompt,
                "metric": metric,
            }
        )

    return Dataset.from_list(records)


def generate(
    output_dir: str | None = None,
    push: bool = False,
    repo_id: str | None = None,
    private: bool = False,
    dry_run: bool = False,
) -> dict[str, list[dict]]:
    """Generate ParallelBench data, optionally saving locally and/or pushing to Hub.

    Args:
        output_dir: Directory to save JSONL files. None to skip local save.
        push: Whether to push to HuggingFace Hub.
        repo_id: HuggingFace repo ID (required when push=True).
        private: Push as private dataset.
        dry_run: Generate in memory without saving or pushing.

    Returns:
        Dict mapping task_name -> list of generated rows.
    """
    import pandas as pd
    from datasets import DatasetDict

    from parallelbench.datasets.task import (
        create_parallel_bench_task,
        task_name_to_config_name,
    )

    split = "test"
    tasks = _load_all_tasks()
    if not tasks:
        console.print("[bold red]Error:[/bold red] No task configs found")
        return {}

    console.print(f"\n[bold]Found {len(tasks)} tasks[/bold]\n")

    all_rows = {}
    failed = []
    sorted_tasks = sorted(tasks.keys())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Generating tasks", total=len(sorted_tasks))

        for task_name in sorted_tasks:
            task_config = copy.deepcopy(tasks[task_name])
            progress.update(overall, description=f"[cyan]{task_name}[/cyan]")

            try:
                rows = create_parallel_bench_task(
                    split=split,
                    task=task_config,
                    output_file=None,
                    no_save=True,
                )
            except Exception as e:
                console.print(f"  [bold yellow]WARN[/bold yellow] {task_name}: {e}")
                failed.append(task_name)
                progress.advance(overall)
                continue

            all_rows[task_name] = rows

            if dry_run:
                progress.advance(overall)
                continue

            if output_dir:
                out_path = Path(output_dir) / split / f"{task_name}.jsonl"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_json(out_path, orient="records", lines=True)

                # Save task_config.yaml alongside JSONL for load_task() compatibility
                out_task_config_file = out_path.parent / "task_config.yaml"
                if out_task_config_file.exists():
                    with open(out_task_config_file, "r") as f:
                        out_task_config = yaml.safe_load(f) or {}
                else:
                    out_task_config = {}
                out_task_config[task_name] = task_config
                with open(out_task_config_file, "w") as f:
                    yaml.dump(out_task_config, f)

            if push:
                config_name = task_name_to_config_name(task_name)
                dataset = _build_hub_dataset(task_name, task_config, rows)
                dataset_dict = DatasetDict({split: dataset})
                dataset_dict.push_to_hub(
                    repo_id,
                    config_name=config_name,
                    private=private,
                )

            progress.advance(overall)

    # Summary
    console.print()
    if failed:
        summary_table = Table(show_header=True, header_style="bold", box=None)
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Count", justify="right")
        summary_table.add_row("[green]Generated[/green]", str(len(all_rows)))
        summary_table.add_row("[red]Failed[/red]", str(len(failed)))
        console.print(summary_table)
        console.print(f"\n[bold yellow]Failed tasks:[/bold yellow] {', '.join(failed)}")
    else:
        console.print(
            f"[bold green]Done![/bold green] Generated {len(all_rows)} tasks."
        )

    console.print()
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate ParallelBench data and optionally push to HuggingFace Hub."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save generated JSONL files",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push generated data to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="HuggingFace repo ID (required with --push)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as private dataset",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate in memory but do not save or push",
    )
    args = parser.parse_args()

    if args.push and not args.repo_id:
        parser.error("--repo_id is required when --push is specified")

    if not args.output_dir and not args.push and not args.dry_run:
        parser.error("Specify --output_dir, --push, or --dry_run")

    generate(
        output_dir=args.output_dir,
        push=args.push,
        repo_id=args.repo_id,
        private=args.private,
        dry_run=args.dry_run,
    )
