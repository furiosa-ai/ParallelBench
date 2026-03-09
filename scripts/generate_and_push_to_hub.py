"""Generate all ParallelBench data from task configs and push directly to HF Hub.

This script does not create local JSONL/output files.

By default, it discovers every config under:
- parallelbench/dataset/data/task_configs/train/*.yaml
- parallelbench/dataset/data/task_configs/test/*.yaml

Then, for each task (e.g. waiting_line/copy), it builds a DatasetDict that
contains all discovered splits (typically train + test) and pushes once.

Usage:
    uv run python scripts/generate_and_push_to_hub.py \
        --repo_id <org/name>
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from parallelbench.datasets.task import (
    create_parallel_bench_task,
    task_name_to_config_name,
)
from parallelbench.datasets.task_utils import load_task_configs


def serialize_column(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def discover_split_config_names() -> dict[str, list[str]]:
    """Return available config names per split from data/task_configs."""
    base_dir = (
        Path(__file__).resolve().parent.parent
        / "parallelbench"
        / "dataset"
        / "data"
        / "task_configs"
    )
    split_to_configs: dict[str, list[str]] = {}

    for split in ("train", "test"):
        split_dir = base_dir / split
        if not split_dir.exists():
            split_to_configs[split] = []
            continue

        config_names = sorted(p.stem for p in split_dir.glob("*.yaml") if p.is_file())
        split_to_configs[split] = config_names

    return split_to_configs


def load_all_tasks() -> dict[str, dict[str, dict]]:
    """Load all tasks keyed by task_name then split.

    Returns:
        {
            "waiting_line/copy": {
                "train": {...task config...},
                "test": {...task config...},
            },
            ...
        }
    """
    split_to_configs = discover_split_config_names()
    tasks_by_name: dict[str, dict[str, dict]] = defaultdict(dict)

    for split, config_names in split_to_configs.items():
        for cfg_name in config_names:
            loaded = load_task_configs(f"{split}/{cfg_name}")
            for task_cfg in loaded.values():
                task_name = task_cfg["name"]
                tasks_by_name[task_name][split] = task_cfg

    return dict(tasks_by_name)


def build_hub_dataset(task_name: str, task_config: dict, rows: list[dict]) -> Dataset:
    prompt = task_config.get("prompt", "")
    metric = task_config.get("metric", "")

    records = []
    for row in rows:
        records.append(
            {
                "input": serialize_column(row["input"]),
                "answer": serialize_column(row["answer"]),
                "output_format": row.get("output_format"),
                "metadata": serialize_column(row["metadata"]),
                "task_name": task_name,
                "prompt": prompt,
                "metric": metric,
            }
        )

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ParallelBench data and push directly to HF Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., org/parallel_bench)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push as private dataset",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate in memory and validate, but do not push",
    )
    args = parser.parse_args()

    all_tasks = load_all_tasks()
    print(f"Loaded {len(all_tasks)} tasks from discovered train/test configs")

    for task_name in sorted(all_tasks.keys()):
        config_name = task_name_to_config_name(task_name)
        split_to_task_cfg = all_tasks[task_name]
        split_datasets = {}

        for split in sorted(split_to_task_cfg.keys()):
            task_config = copy.deepcopy(split_to_task_cfg[split])
            rows = create_parallel_bench_task(
                split=split,
                task=task_config,
                output_file=None,
                no_save=True,
            )
            dataset = build_hub_dataset(task_name, task_config, rows)
            split_datasets[split] = dataset
            print(f"Prepared {config_name} ({split}, {len(dataset)} samples)")

        dataset_dict = DatasetDict(split_datasets)
        if args.dry_run:
            continue

        dataset_dict.push_to_hub(
            args.repo_id,
            config_name=config_name,
            private=args.private,
        )
        print(
            f"Pushed {config_name} with splits: {', '.join(sorted(split_datasets.keys()))}"
        )


if __name__ == "__main__":
    main()
