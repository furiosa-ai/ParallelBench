from pathlib import Path

from parallelbench.dataset.task import (
    PARALLEL_BENCH_MASK_TOKEN as PARALLEL_BENCH_MASK_TOKEN,
    create_parallel_bench_task as create_parallel_bench_task,
    load_task,
    task_name_to_config_name as task_name_to_config_name,
    config_name_to_task_name as config_name_to_task_name,
)
from parallelbench.dataset.metrics import Metric, parallel_bench_metric_func_map
from parallelbench.dataset.task_utils import load_task_configs


def get_task_names(split="test"):
    config_dir = Path(__file__).parent / "data" / "task_configs" / split
    task_names = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        tasks = load_task_configs(str(yaml_file))
        task_names.extend(tasks.keys())
    task_names = sorted(t for t in task_names if not t.startswith("_"))
    return task_names


PARALLEL_BENCH_TASKS = get_task_names()


class ParallelBench:
    def __init__(
        self, task, split="test", num_samples=None, infill=False, from_hub=None
    ):
        self.task = task
        self.infill = infill
        self.ds, task_config = load_task(split, task, from_hub=from_hub)

        self.prompt = task_config["prompt"]
        self.metric_name = task_config["metric"]
        metric_func = parallel_bench_metric_func_map[task_config["metric"]]
        if isinstance(metric_func, type) and issubclass(metric_func, Metric):
            self.metric_func = metric_func()
        else:
            self.metric_func = metric_func

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
            messages_icl = [
                [
                    {
                        "role": "user",
                        "content": self.prompt.format(**icl_example["input"]).replace(
                            "\\n", "\n"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": icl_example["answer"]
                        if isinstance(icl_example["answer"], str)
                        else icl_example["answer"]["example"],
                    },
                ]
                for icl_example in sample["input"]["icl_examples"]
            ]

            messages_icl = [item for sublist in messages_icl for item in sublist]
            messages = messages_icl + messages

        input = dict(messages=messages)

        if self.infill:
            input["output_prefix"] = sample["output_format"]

        return dict(
            input=input, label=sample["answer"], index=idx, metadata=sample["metadata"]
        )

    def compute_metrics(
        self, predictions, references, output_per_sample=False, **kwargs
    ):
        assert len(predictions) == len(references), (
            "Predictions and references must have the same length."
        )

        metrics_per_sample = []
        for pred, ref in zip(predictions, references):
            metrics_per_sample.append(self.metric_func(pred, ref))

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
                "answer": label
                if isinstance(label, str)
                else label.get("example", label.get("result")),
                "task": self.task,
            }
