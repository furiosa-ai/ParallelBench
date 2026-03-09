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

import pandas as pd
from datasets import Dataset, DatasetDict

from parallelbench.datasets.task import (
    create_parallel_bench_task,
    task_name_to_config_name,
)
from parallelbench.datasets.task_utils import load_task_configs


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
    config_dir = Path(__file__).resolve().parent / "data" / "task_configs" / "test"
    if not config_dir.exists():
        return {}

    tasks = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        loaded = load_task_configs(str(yaml_file))
        tasks.update(loaded)
    return tasks


def _build_hub_dataset(task_name: str, task_config: dict, rows: list[dict]) -> Dataset:
    """Convert generated rows into a Hub-ready Dataset with serialized columns."""
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
    split = "test"
    tasks = _load_all_tasks()
    if not tasks:
        print("No task configs found")
        return {}

    print(f"Found {len(tasks)} tasks")

    all_rows = {}
    failed = []
    for task_name in sorted(tasks.keys()):
        task_config = copy.deepcopy(tasks[task_name])

        try:
            rows = create_parallel_bench_task(
                split=split,
                task=task_config,
                output_file=None,
                no_save=True,
            )
        except Exception as e:
            print(f"  [WARN] {task_name} failed: {e}")
            failed.append(task_name)
            continue

        all_rows[task_name] = rows
        print(f"  Generated {task_name} ({len(rows)} samples)")

        if dry_run:
            continue

        if output_dir:
            out_path = Path(output_dir) / split / f"{task_name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_json(out_path, orient="records", lines=True)
            print(f"    Saved to {out_path}")

        if push:
            config_name = task_name_to_config_name(task_name)
            dataset = _build_hub_dataset(task_name, task_config, rows)
            dataset_dict = DatasetDict({split: dataset})
            dataset_dict.push_to_hub(
                repo_id,
                config_name=config_name,
                private=private,
            )
            print(f"    Pushed to {repo_id} (config={config_name})")

    if failed:
        print(f"Done! Generated {len(all_rows)} tasks, {len(failed)} failed: {failed}")
    else:
        print(f"Done! Generated {len(all_rows)} tasks.")
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
