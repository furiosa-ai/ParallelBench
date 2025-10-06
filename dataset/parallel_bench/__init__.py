from pathlib import Path
import pandas as pd
import yaml

from dataset.parallel_bench.data.task import create_parallel_bench_task, load_task
from dataset.parallel_bench.metrics import Metric, parallel_bench_metric_func_map

from datasets import Dataset


# def load_dataset(path, name, split):
#     dataset_file = Path(__file__).parent / "data" / "output" / f"{name}.jsonl"

#     if not dataset_file.exists():
#         create_parallel_bench_task(name, dataset_file)

#     return Dataset.from_pandas(pd.read_json(path_or_buf=dataset_file, lines=True))


def get_task_names(split="test"):
    path = Path(__file__).parent / "data" / "output" / split
    task_names = sorted((str(p.relative_to(path)).rsplit(".", 1)[0]) for p in path.glob("*/*.jsonl"))
    task_names = [t for t in task_names if not t[0] == "_"]
    return task_names


PARALLEL_BENCH_TASKS = get_task_names()


class ParallelBench:
    def __init__(self, task, split="test", num_samples=None, infill=False):
        # self.ds = load_dataset("parallel_bench", task, split="test")

        # with open(Path(__file__).parent / "data" / "task_config.yaml", "r") as f:
        #     task_config = yaml.safe_load(f)[task]

        self.task = task
        self.infill = infill
        self.ds, task_config = load_task(split, task)

        self.prompt = task_config["prompt"]
        self.metric_name = task_config["metric"]
        self.metric_func = parallel_bench_metric_func_map[task_config["metric"]]

        try:
            if issubclass(self.metric_func, Metric):
                self.metric_func = self.metric_func()
        except TypeError:
            pass

        if num_samples is not None:
            self.ds = self.ds.select([i for i in list(range(num_samples))])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        messages = [
            {
                "role": "user",
                "content": self.prompt.format(**sample["input"]).replace("\\n", "\n"),
            }
        ]

        if "icl_examples" in sample["input"]:
            messages_icl = [[
                {
                    "role": "user",
                    "content": self.prompt.format(**icl_example["input"]).replace("\\n", "\n"),
                },
                {
                    "role": "assistant",
                    "content": icl_example["answer"] if isinstance(icl_example["answer"], str) else icl_example["answer"]["example"],
                }
            ] for icl_example in sample["input"]["icl_examples"]]

            messages_icl = [item for sublist in messages_icl for item in sublist]
            messages = messages_icl + messages

        input = dict(messages=messages)

        if self.infill:
            input["output_prefix"] = sample["output_format"]

        return dict(input=input, label=sample["answer"], index=idx, metadata=sample["metadata"])

    def compute_metrics(self, predictions, references, output_per_sample=False, **kwargs):
        assert len(predictions) == len(references), "Predictions and references must have the same length."

        score = 0
        score_strict = 0

        metrics_per_sample = []
        for pred, ref in zip(predictions, references):
            sample_metrics = self.metric_func(pred, ref)

            if isinstance(sample_metrics, float):
                score = sample_metrics
                score_strict = self.metric_func(pred, ref, strict=True)
                metrics_per_sample.append({
                    "score": score,
                    "score_strict": score_strict
                })
            elif isinstance(sample_metrics, dict):
                metrics_per_sample.append(sample_metrics)

        metric_names = list(metrics_per_sample[0].keys())
        metrics = {name: [m[name] for m in metrics_per_sample] for name in metric_names}
        for name in metric_names:
            metrics[name] = sum(metrics[name]) / len(metrics[name]) * 100.0

        if output_per_sample:
            return metrics, metrics_per_sample
        else:
            return metrics
    
    def to_sft_dataset(self):
        for i in range(len(self.ds)):
            sample = self[i]

            prompt = sample["input"]["messages"][-1]["content"]
            label = sample["label"]

            yield {
                "prompt": prompt,
                "answer": label if isinstance(label, str) else label.get("example", label.get("result")),
                "task": self.task,
            }
