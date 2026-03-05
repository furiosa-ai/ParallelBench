"""Task loading, creation, and CLI for ParallelBench benchmark.

This module provides:
- load_task(): Load pre-generated task data from JSONL files
- create_parallel_bench_task(): Generate and save task data
- CLI entry point for batch task creation
"""

from pathlib import Path
import argparse
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
    """config_name의 첫 번째 "-" → "/"로 변환합니다 (e.g., "waiting_line-copy" → "waiting_line/copy")."""
    # 카테고리와 task 사이의 구분자는 첫 번째 "-"이지만,
    # 카테고리 자체에 "_"가 포함될 수 있으므로 단순 replace 불가.
    # Hub config name은 "category-task" 형식이므로 첫 번째 "-"를 "/"로 변환.
    return config_name.replace("-", "/", 1)


def _load_task_from_hub(repo_id, split, task_name):
    """HuggingFace Hub에서 task를 로드합니다."""
    config_name = task_name_to_config_name(task_name)
    hub_dataset = load_dataset(repo_id, config_name, split=split)

    # 첫 row에서 prompt/metric 추출하여 task_config 구성
    first_row = hub_dataset[0]
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
    assert num_samples % samples_per_length == 0, (
        "num_samples must be divisible by samples_per_length"
    )
    num_buckets = num_samples // samples_per_length

    data_per_length = {}

    finished = False
    for sample in tqdm(generate_parallel_bench_task_random(rng, task, infinite=True)):
        length = (
            sample["metadata"]["length"]
            if task["type"] != "repeat"
            else sample["metadata"]["count"]
        )
        if length not in data_per_length:
            data_per_length[length] = []

        if len(data_per_length[length]) < samples_per_length:
            data_per_length[length].append(sample)

        if (
            sum(len(data) == samples_per_length for data in data_per_length.values())
            == num_buckets
        ):
            data_per_length = {
                k: v for k, v in data_per_length.items() if len(v) == samples_per_length
            }
            finished = True
            break

    assert finished
    lengths = sorted(data_per_length.keys())
    data = sum([data_per_length[length] for length in lengths], [])
    assert len(data) == num_samples, f"Expected {num_samples} samples, got {len(data)}"

    return data


def _create_task(rng, task):
    print(f"Creating task {task['name']} with seed {task['seed']}...")
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
            for t in range(task["icl_example_count"])
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
            assert False

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
