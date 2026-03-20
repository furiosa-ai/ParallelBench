"""ParallelBench task implementation for lm-eval-harness.

Each ParallelBench task (e.g. waiting_line/copy, puzzles/sudoku_n4) is defined
by a YAML file that sets metadata.parallelbench_task to the task identifier.
This class loads data from the existing ParallelBench dataset, formats prompts,
and computes metrics using the existing metric functions.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict
from lm_eval.api.task import ConfigurableTask

from parallelbench.datasets import ParallelBench
from parallelbench.datasets.metrics import Metric
from parallelbench.lm_eval_wrappers.metadata_store import MetadataStore
from parallelbench.tasks.utils import (
    METADATA_METRIC_KEYS,
    compute_sample_metrics,
    get_metric_keys_for_task,
)


class ParallelBenchTask(ConfigurableTask):
    """lm-eval task wrapping a single ParallelBench evaluation task.

    YAML config must include:
        metadata:
          parallelbench_task: "waiting_line/copy"  # dataset task identifier

    Hub에서 데이터를 로드하려면 dataset_path를 설정합니다:
        dataset_path: "furiosa-ai/ParallelBench"

    The task loads data from ParallelBench, uses the existing prompt templates,
    and computes metrics using the existing metric functions. dLLM metadata
    (NFE, decoding order) is extracted from MetadataStore.
    """

    VERSION = 0.1
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None, **kwargs):
        if config and "class" in config:
            config = {k: v for k, v in config.items() if k != "class"}

        # Initialize ParallelBench before super().__init__ because the parent
        # calls fewshot_docs() -> test_docs() during init, which needs _parallelbench.
        metadata = (config or {}).get("metadata", {})
        task_name = metadata.get("parallelbench_task")
        if task_name is None:
            raise ValueError(
                "ParallelBenchTask requires metadata.parallelbench_task in YAML config"
            )

        flex_config = metadata.get("flex_config")
        from_hub = (config or {}).get("dataset_path")
        # dataset_path (Hub) takes priority over flex_config so that
        # adding dataset_path to a flex-enabled YAML switches to paper-reproduce mode.
        if from_hub:
            flex_config = None
        self._parallelbench = ParallelBench(
            task=task_name, split="test", from_hub=from_hub, flex_config=flex_config
        )
        self._metric_name = self._parallelbench.metric_name

        metric_func = self._parallelbench.metric_func
        if isinstance(metric_func, type) and issubclass(metric_func, Metric):
            metric_func = metric_func()
        self._metric_func = metric_func

        self._metric_keys = get_metric_keys_for_task(self._metric_name)
        self._all_metric_keys = self._metric_keys + METADATA_METRIC_KEYS

        super().__init__(config=config, **kwargs)

    def fewshot_docs(self):
        """Provide ICL examples from task config as fewshot documents.

        lm-eval's ContextSampler uses these docs to build fewshot context
        via doc_to_text() and doc_to_target(). When --num_fewshot 1 is used
        with --apply_chat_template --fewshot_as_multiturn, the sampler builds
        multi-turn chat history from these examples.

        Returns an empty list for tasks without ICL examples (e.g., text_writing).
        """
        if len(self._parallelbench) == 0:
            return []

        first_sample = self._parallelbench.ds[0]
        input_data = first_sample.get("input", {})
        if isinstance(input_data, str):
            import json

            try:
                input_data = json.loads(input_data)
            except (json.JSONDecodeError, ValueError):
                return []

        icl_examples = input_data.get("icl_examples")
        if not icl_examples:
            return []

        prompt = self._parallelbench.prompt
        docs = []
        for icl_example in icl_examples:
            question = prompt.format(**icl_example["input"]).replace("\\n", "\n")
            answer = icl_example["answer"]
            if isinstance(answer, dict):
                answer = answer.get("example", answer.get("result", str(answer)))
            docs.append(
                {
                    "messages": [{"role": "user", "content": question}],
                    "label": answer,
                }
            )
        return docs

    def download(self, dataset_kwargs=None) -> None:
        """Build a HuggingFace DatasetDict from local ParallelBench data."""
        docs = []
        for idx in range(len(self._parallelbench)):
            sample = self._parallelbench[idx]
            docs.append(
                {
                    "messages": sample["input"]["messages"],
                    "label": sample["label"],
                    "metadata": sample["metadata"],
                    "index": sample["index"],
                    "output_prefix": sample["input"].get("output_prefix"),
                }
            )
        self.dataset = DatasetDict({"test": Dataset.from_list(docs)})

    def doc_to_text(self, doc: dict) -> str:
        """Return the prompt text from the last user message."""
        messages = doc["messages"]
        return messages[-1]["content"]

    def doc_to_target(self, doc: dict) -> str:
        """Return the reference answer as a string."""
        label = doc["label"]
        if isinstance(label, str):
            return label
        if isinstance(label, dict):
            return label.get("example", label.get("result", str(label)))
        return str(label)

    def process_results(self, doc: dict, results: list[str]) -> dict:
        """Compute metrics for a single sample.

        Combines ParallelBench task metrics with dLLM metadata from MetadataStore.
        """
        prediction = results[0]
        reference = doc["label"]

        sample_metrics = compute_sample_metrics(
            metric_name=self._metric_name,
            prediction=prediction,
            reference=reference,
            metric_func=self._metric_func,
        )

        metadata = MetadataStore.instance().pop()

        sample_metrics["nfe"] = float(metadata.nfe)
        sample_metrics["tokens_per_step"] = (
            float(metadata.tokens_per_step)
            if metadata.tokens_per_step is not None
            else 0.0
        )
        sample_metrics["input_length"] = float(metadata.input_length or 0)
        sample_metrics["output_length"] = float(metadata.output_length or 0)

        return sample_metrics

    def aggregation(self) -> dict:
        """Return aggregation functions for each metric key.

        Task metrics are scaled to percentage (x100). Metadata metrics use plain mean.
        """
        agg = {}
        for key in self._metric_keys:
            agg[key] = _mean_percentage
        for key in METADATA_METRIC_KEYS:
            agg[key] = np.mean
        return agg

    def higher_is_better(self) -> dict:
        result = {}
        for key in self._metric_keys:
            result[key] = True
        result["nfe"] = False
        result["tokens_per_step"] = True
        result["input_length"] = None
        result["output_length"] = None
        return result


def _mean_percentage(items: list[float]) -> float:
    """Compute mean and scale to percentage."""
    if not items:
        return 0.0
    return sum(items) / len(items) * 100.0
