"""Task loading, creation, and CLI for ParallelBench benchmark.

This module provides:
- load_task(): Load pre-generated task data from JSONL files
- create_parallel_bench_task(): Generate and save task data
- CLI entry point for batch task creation
"""

from pathlib import Path
import argparse
from collections import Counter
import json
import random

import pandas as pd
from tqdm import tqdm
import yaml

from datasets import Dataset, load_dataset

from parallelbench.dataset.task_utils import (
    _get_task_file,
    load_task_configs,
    load_words_from_file,
    str_to_seed,
)
from parallelbench.dataset.task_generators import TASK_GENERATORS

# Ensure all generators are registered by importing the module


DEFAULT_SEED = 42


PARALLEL_BENCH_MASK_TOKEN = "[MASK]"


def _try_json_loads(value):
    """JSON 파싱을 시도하고, 실패하면 원본 string을 반환합니다."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def task_name_to_config_name(task_name: str) -> str:
    """task_name의 "/" → "-"로 변환합니다 (Hub config naming convention)."""
    return task_name.replace("/", "-")


def config_name_to_task_name(config_name: str) -> str:
    """config_name의 첫 번째 "-" → "/"로 변환합니다 (e.g., "waiting_line-copy" → "waiting_line/copy").

    Convention: category names use underscores (e.g., "waiting_line", "text_writing"),
    and the first hyphen is always the category/task separator. Task names within a
    category must not contain hyphens to ensure correct roundtrip conversion.
    """
    return config_name.replace("-", "/", 1)


def _load_task_from_hub(repo_id, split, task_name):
    """HuggingFace Hub에서 task를 로드합니다."""
    config_name = task_name_to_config_name(task_name)
    hub_dataset = load_dataset(repo_id, config_name, split=split)

    # 첫 row에서 prompt/metric 추출하여 task_config 구성
    first_row = hub_dataset[0]
    for required_key in ("prompt", "metric"):
        if required_key not in first_row:
            raise ValueError(
                f"Hub dataset '{repo_id}' (config='{config_name}') is missing "
                f"required column '{required_key}'. "
                f"Available columns: {list(first_row.keys())}"
            )
    task_config = {
        "prompt": first_row["prompt"],
        "metric": first_row["metric"],
    }

    # JSON string 컬럼 deserialize, config 컬럼 제거
    records = []
    for row in hub_dataset:
        records.append(
            {
                "input": _try_json_loads(row["input"]),
                "answer": _try_json_loads(row["answer"]),
                "output_format": row.get("output_format"),
                "metadata": _try_json_loads(row["metadata"]),
            }
        )

    task = Dataset.from_list(records)
    return task, task_config


def load_task(split, task_name, from_hub=None):
    if from_hub is not None:
        return _load_task_from_hub(from_hub, split, task_name)

    task_file = _get_task_file(split, task_name)
    task_config_file = task_file.parent / "task_config.yaml"

    with open(task_config_file, "r") as f:
        task_config = yaml.safe_load(f)

    task_config = task_config[task_name]
    task = Dataset.from_pandas(pd.read_json(path_or_buf=task_file, lines=True))

    return task, task_config


def generate_parallel_bench_task_random(rng, task_config, infinite=False):
    task_config = {**task_config}

    if infinite:
        task_config["num_samples"] = int(1e10)

    task_type = task_config["type"]
    generator = TASK_GENERATORS.get(task_type)
    if generator is None:
        raise ValueError(f"Unknown task type: {task_type}")
    yield from generator(rng, task_config)


def create_parallel_bench_task_random(rng, task):
    return list(generate_parallel_bench_task_random(rng, task))


def create_parallel_bench_task_random_samples_per_length(rng, task):
    samples_per_length = task["samples_per_length"]
    num_samples = task["num_samples"]
    if num_samples % samples_per_length != 0:
        raise ValueError("num_samples must be divisible by samples_per_length")
    num_buckets = num_samples // samples_per_length

    bucket_values = None
    if task["type"] == "repeat" and "repeat_counts" in task:
        bucket_values = list(task["repeat_counts"])
    elif "lengths" in task and task["lengths"]:
        bucket_values = list(task["lengths"])
    elif "min_length" in task and "max_length" in task:
        bucket_values = list(range(task["min_length"], task["max_length"] + 1))
    elif "size" in task:
        bucket_values = [task["size"]]

    target_counts = None
    if bucket_values is not None:
        target_counts = {
            k: v * samples_per_length for k, v in Counter(bucket_values).items()
        }
        if sum(target_counts.values()) != num_samples:
            # 고정 길이/고정 카운트 task의 경우 전체 샘플 수를 그대로 생성하도록 보정
            if len(target_counts) == 1:
                only_key = next(iter(target_counts.keys()))
                target_counts = {only_key: num_samples}
            else:
                raise ValueError(
                    f"Inconsistent samples_per_length config for task={task['name']}: "
                    f"expected {num_samples}, got {sum(target_counts.values())} from inferred buckets."
                )

    data_per_length = {}
    max_steps = max(100000, num_samples * 500)

    finished = False
    for i, sample in enumerate(
        tqdm(generate_parallel_bench_task_random(rng, task, infinite=True)), start=1
    ):
        length = (
            sample["metadata"]["length"]
            if task["type"] != "repeat"
            else sample["metadata"]["count"]
        )
        if length not in data_per_length:
            data_per_length[length] = []

        target_count = (
            target_counts.get(length, 0)
            if target_counts is not None
            else samples_per_length
        )
        if len(data_per_length[length]) < target_count:
            data_per_length[length].append(sample)

        if target_counts is not None:
            if all(
                len(data_per_length.get(k, [])) >= v for k, v in target_counts.items()
            ):
                finished = True
                break
        else:
            if (
                sum(
                    len(data) == samples_per_length for data in data_per_length.values()
                )
                == num_buckets
            ):
                data_per_length = {
                    k: v
                    for k, v in data_per_length.items()
                    if len(v) == samples_per_length
                }
                finished = True
                break

        if i >= max_steps:
            raise RuntimeError(
                f"Task generation timed out for task={task['name']} with samples_per_length={samples_per_length}. "
                "Please check task config (length buckets may be unreachable)."
            )

    if not finished:
        raise RuntimeError(
            f"Task generation did not finish for task={task['name']}. "
            "Check task config for unreachable constraints."
        )
    if target_counts is not None:
        lengths = sorted(target_counts.keys())
        data = sum(
            [data_per_length[length][: target_counts[length]] for length in lengths], []
        )
    else:
        lengths = sorted(data_per_length.keys())
        data = sum([data_per_length[length] for length in lengths], [])
    if len(data) != num_samples:
        raise RuntimeError(f"Expected {num_samples} samples, got {len(data)}")

    return data


def _create_task(rng, task):
    import logging

    logging.getLogger(__name__).info(
        f"Creating task {task['name']} with seed {task['seed']}..."
    )
    if task.get("samples_per_length", 0) > 0:
        data = create_parallel_bench_task_random_samples_per_length(rng, task)
    else:
        data = create_parallel_bench_task_random(rng, task)
    return data


def create_parallel_bench_task(split, task, output_file, rng=None, no_save=False):
    if not output_file:
        output_file = _get_task_file(split, task_name=task["name"])

    if "seed" not in task:
        task["seed"] = str_to_seed(task["name"].split("/")[-1], DEFAULT_SEED)
    else:
        task["seed"] = str_to_seed(task["name"].split("/")[-1], task["seed"])

    if rng is None:
        rng = random.Random(task["seed"])

    if "words" in task:
        words_file = task["words"]
        task["words"] = load_words_from_file(task["words"])
    else:
        words_file = None

    data = _create_task(rng, task)

    if task.get("icl_example_count", 0) > 0:
        icl_datasets = [
            create_parallel_bench_task_random(
                rng=rng, task={**task, "icl_example_count": 0}
            )
            for _ in range(task["icl_example_count"])
        ]

        for i, sample in enumerate(data):
            sample["input"]["icl_examples"] = [
                icl_dataset[i] for icl_dataset in icl_datasets
            ]

    if not no_save:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        data.to_json(output_file, orient="records", lines=True)

        out_task_config_file = Path(output_file).parent / "task_config.yaml"

        if out_task_config_file.exists():
            with open(out_task_config_file, "r") as f:
                out_task_config = yaml.safe_load(f)
        else:
            out_task_config = {}

        out_task_config[task["name"]] = task

        if words_file is not None:
            task["words"] = words_file

        with open(out_task_config_file, "w") as f:
            yaml.dump(out_task_config, f)
    else:
        return data


def main(task, **kwargs):
    loaded_tasks = []

    for t in task:
        if t.endswith("/all"):
            t = t[: -len("/all")]
            split = t.split("/")[0]
            tasks = list(load_task_configs(t).values())
            loaded_tasks.extend(list(zip([split] * len(tasks), tasks)))
        else:
            raise ValueError(
                f"Unsupported task format: '{t}'. Expected format: '<split>/all'"
            )

    for split, t in loaded_tasks:
        create_parallel_bench_task(split=split, task=t, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a DLLM task.")
    parser.add_argument(
        "--task", type=str, nargs="+", required=True, help="Name of the task to create."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Output file to save the task data.",
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())
