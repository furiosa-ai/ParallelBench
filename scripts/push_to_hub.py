"""Push ParallelBench dataset to HuggingFace Hub.

각 task를 별도의 HF config으로 push합니다.
Config 이름은 task_name의 "/" → "--"로 변환합니다 (e.g., "waiting_line/copy" → "waiting_line--copy").

Usage:
    uv run python scripts/push_to_hub.py --repo_id <org/name> [--private] [--split test]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict


DATA_DIR = (
    Path(__file__).parent.parent / "parallelbench" / "dataset" / "data" / "output"
)


def task_name_to_config_name(task_name: str) -> str:
    """task_name의 "/" → "-"로 변환합니다 (e.g., "waiting_line/copy" → "waiting_line-copy")."""
    return task_name.replace("/", "-")


def serialize_column(value):
    """dict/list는 JSON serialize하고, string은 그대로 반환합니다."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_all_tasks(split: str) -> list[tuple[str, dict, Path]]:
    """지정된 split의 모든 task를 (task_name, task_config, jsonl_path) 튜플 리스트로 반환합니다."""
    split_dir = DATA_DIR / split
    tasks = []

    for task_config_path in sorted(split_dir.glob("*/task_config.yaml")):
        with open(task_config_path) as f:
            all_configs = yaml.safe_load(f)

        category_dir = task_config_path.parent
        for task_name, task_config in all_configs.items():
            jsonl_name = task_name.split("/")[-1]
            jsonl_path = category_dir / f"{jsonl_name}.jsonl"
            if jsonl_path.exists():
                tasks.append((task_name, task_config, jsonl_path))

    return tasks


def build_hub_dataset(task_name: str, task_config: dict, jsonl_path: Path) -> Dataset:
    """단일 task의 JSONL + config을 Hub용 Dataset으로 변환합니다."""
    df = pd.read_json(jsonl_path, lines=True)
    records = []

    prompt = task_config.get("prompt", "")
    metric = task_config.get("metric", "")

    for _, row in df.iterrows():
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


# (로컬 task_name, Hub에서 사용할 task_name) 매핑
# 로컬 task_name → Hub task_name 매핑
BENCHMARK_TASKS = {
    "waiting_line/copy": "waiting_line/copy",
    "waiting_line/sort": "waiting_line/sort",
    "waiting_line/reverse": "waiting_line/reverse",
    "waiting_line/shuffle": "waiting_line/shuffle",
    "waiting_line/replace_index": "waiting_line/replace_index",
    "waiting_line/replace_random": "waiting_line/replace_random",
    "waiting_line/insert_index": "waiting_line/insert_index",
    "waiting_line/insert_random": "waiting_line/insert_random",
    "waiting_line/remove_index": "waiting_line/remove_index",
    "waiting_line/remove_random": "waiting_line/remove_random",
    "paraphrase_summarize/samsum": "text_writing/summarization",
    "paraphrase_summarize/chatgpt-paraphrases": "text_writing/paraphrasing",
    "words_to_sentence/easy": "text_writing/words_to_sentence_easy",
    "words_to_sentence/medium": "text_writing/words_to_sentence_medium",
    "words_to_sentence/hard": "text_writing/words_to_sentence_hard",
    "puzzle/latin_square_n4": "puzzles/latin_square_n4",
    "puzzle/sudoku_n4_12": "puzzles/sudoku_n4_12",
}


def main():
    parser = argparse.ArgumentParser(
        description="Push ParallelBench to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., org/parallel_bench)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Push as private dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split to push (default: test)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Push all tasks (default: benchmark 17 tasks only)",
    )
    args = parser.parse_args()

    tasks = load_all_tasks(args.split)

    if not args.all:
        tasks = [
            (name, cfg, path) for name, cfg, path in tasks if name in BENCHMARK_TASKS
        ]

    print(f"Found {len(tasks)} tasks in '{args.split}' split")

    for task_name, task_config, jsonl_path in tasks:
        hub_task_name = BENCHMARK_TASKS.get(task_name, task_name)
        config_name = task_name_to_config_name(hub_task_name)
        dataset = build_hub_dataset(hub_task_name, task_config, jsonl_path)
        dataset_dict = DatasetDict({args.split: dataset})

        print(f"  Pushing {config_name} ({len(dataset)} samples)...")
        dataset_dict.push_to_hub(
            args.repo_id,
            config_name=config_name,
            private=args.private,
        )

    print(f"Done! Pushed {len(tasks)} configs to {args.repo_id}")


if __name__ == "__main__":
    main()
