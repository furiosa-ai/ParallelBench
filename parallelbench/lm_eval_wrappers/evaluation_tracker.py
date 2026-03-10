"""Custom EvaluationTracker that organizes results by structured subdirectories.

lm-eval's default output structure:
    {output_path}/{model_sanitized}/results_{timestamp}.json

ParallelBench output structure:
    {output_path}/{model_sanitized}/{remasking}/{repr_param_value}/{run_id}/results_{task_name}.json

The repr_param_value encodes the representative parameter for the remasking strategy:
    - topk strategies: "tps{tokens_per_step}" (e.g., tps4)
    - threshold strategies: "t{alg_threshold}" (e.g., t0.3)
    - factor strategies: "f{alg_factor}" (e.g., f2.0)

This allows browsing results by (model, remasking, repr_param) tuple and comparing
across tasks and hyperparameter sweeps.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from lm_eval.loggers.evaluation_tracker import EvaluationTracker, sanitize_list
from lm_eval.utils import handle_non_serializable, hash_string

logger = logging.getLogger(__name__)


def build_gen_kwargs_dirname(gen_kwargs: dict | None) -> str:
    """Build a directory name from gen_kwargs dict (excluding remasking).

    Produces a compact, filesystem-safe string like 'bl32_s32'.
    Keys are abbreviated and ordered for readability:
        block_length -> bl, steps -> s, max_tokens -> mt,
        alg_threshold -> at, alg_factor -> af, temperature -> t
    Unknown keys are appended as key=value pairs sorted alphabetically.

    Remasking is excluded because it gets its own directory level.

    Returns 'default' when gen_kwargs is empty or None (after removing remasking).

    .. deprecated::
        Use _resolve_repr_param_value() with the unmasking registry instead.
        Kept as fallback for backward compatibility.
    """
    if not gen_kwargs:
        return "default"

    abbreviations = {
        "block_length": "bl",
        "steps": "s",
        "max_tokens": "mt",
        "alg_threshold": "at",
        "alg_factor": "af",
        "temperature": "t",
        "alg_temp": "algt",
    }

    # Ordered keys for consistent, readable directory names
    ordered_keys = [
        "block_length",
        "steps",
        "max_tokens",
        "temperature",
        "alg_temp",
        "alg_threshold",
        "alg_factor",
    ]

    parts = []
    used_keys = set()

    for key in ordered_keys:
        if key not in gen_kwargs:
            continue
        used_keys.add(key)
        value = gen_kwargs[key]
        abbr = abbreviations.get(key, key)

        # Format numeric values: drop trailing .0 for clean names
        if isinstance(value, float) and value == int(value):
            value = int(value)

        parts.append(f"{abbr}{value}")

    # Append any unknown keys sorted alphabetically
    for key in sorted(gen_kwargs.keys()):
        if key in used_keys:
            continue
        # Skip lm-eval internal keys and remasking (separate directory level)
        if key in ("until", "do_sample", "remasking"):
            continue
        value = gen_kwargs[key]
        if isinstance(value, float) and value == int(value):
            value = int(value)
        parts.append(f"{key}{value}")

    return "_".join(parts) if parts else "default"


def extract_remasking(gen_kwargs: dict | None) -> str:
    """Extract remasking strategy from gen_kwargs.

    Returns 'default' when gen_kwargs is empty or remasking is not specified.
    """
    if not gen_kwargs:
        return "default"
    return str(gen_kwargs.get("remasking", "default"))


def extract_task_name(results: dict) -> str:
    """Extract task name from results dict.

    lm-eval stores results keyed by task name. For single-task runs,
    there is exactly one key. For multi-task runs, uses the common prefix
    (e.g., "parallel_bench") to avoid overly long filenames.
    Falls back to 'unknown_task' if empty.
    """
    task_names = list(results.get("results", {}).keys())
    if len(task_names) == 1:
        return task_names[0]
    if task_names:
        # Find common prefix to produce a short group name
        prefix = os.path.commonprefix(sorted(task_names)).rstrip("_")
        if prefix:
            return prefix
        return f"{len(task_names)}_tasks"
    return "unknown_task"


def _resolve_repr_param_value(gen_kwargs: dict | None) -> str:
    """Compute the representative parameter value directory segment.

    Uses the unmasking registry to determine the strategy type and
    format the representative parameter accordingly:
      - topk: "tps{int(max_tokens / steps)}" (e.g., tps4)
      - threshold: "t{alg_threshold}" (e.g., t0.3)
      - factor: "f{alg_factor}" (e.g., f2.0)

    Falls back to build_gen_kwargs_dirname() if the registry lookup fails.
    """
    if not gen_kwargs:
        return build_gen_kwargs_dirname(gen_kwargs)

    remasking = gen_kwargs.get("remasking")
    if not remasking:
        return build_gen_kwargs_dirname(gen_kwargs)

    try:
        from parallelbench.models.unmasking_registry import get_strategy_type

        strategy_type = get_strategy_type(remasking)
    except (KeyError, ImportError):
        return build_gen_kwargs_dirname(gen_kwargs)

    if strategy_type == "topk":
        max_tokens = gen_kwargs.get("max_tokens")
        steps = gen_kwargs.get("steps")
        if max_tokens is not None and steps is not None and steps != 0:
            tps_value = int(max_tokens / steps)
            return f"tps{tps_value}"
        return build_gen_kwargs_dirname(gen_kwargs)

    if strategy_type == "threshold":
        alg_threshold = gen_kwargs.get("alg_threshold")
        if alg_threshold is not None:
            return f"t{alg_threshold}"
        return build_gen_kwargs_dirname(gen_kwargs)

    if strategy_type == "factor":
        alg_factor = gen_kwargs.get("alg_factor")
        if alg_factor is not None:
            return f"f{alg_factor}"
        return build_gen_kwargs_dirname(gen_kwargs)

    return build_gen_kwargs_dirname(gen_kwargs)


class ParallelBenchEvaluationTracker(EvaluationTracker):
    """EvaluationTracker subclass that adds structured subdirectory to output path.

    Overrides save_results_aggregated and save_results_samples to insert
    a repr_param_value/run_id subdirectory between the model/remasking directories
    and result files.

    Class attributes:
        run_id: Set by parallelbench/cli/eval.py before instantiation.
                Either user-provided via --run_id or an auto-generated 8-char hex UUID.
    """

    run_id: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will be set by save_results_aggregated for reuse in save_results_samples
        self._gen_kwargs_output_dir: Path | None = None

    def _resolve_output_dir(
        self, gen_kwargs: dict | None, results: dict | None = None
    ) -> Path:
        """Build the output directory: model/remasking/repr_param_value/run_id."""
        path = Path(self.output_path if self.output_path else Path.cwd())
        path = path / self.general_config_tracker.model_name_sanitized
        path = path / extract_remasking(gen_kwargs)
        path = path / _resolve_repr_param_value(gen_kwargs)
        run_id = self.__class__.run_id or "unknown"
        path = path / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict | None = None,
    ) -> None:
        """Save aggregated results with structured subdirectory."""
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                # Compute task hashes inline (matches upstream lm-eval logic)
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [
                            s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                            for s in task_samples
                        ]
                        task_hashes[task_name] = hash_string("".join(sample_hashes))

                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )

                gen_kwargs = results.get("config", {}).get("gen_kwargs")
                path = self._resolve_output_dir(gen_kwargs, results)
                self._gen_kwargs_output_dir = path

                task_name = extract_task_name(results)
                file_path = path / f"results_{task_name}.json"
                file_path.open("w", encoding="utf-8").write(dumped)

                logger.info(
                    "Results saved to %s",
                    file_path,
                )

                # Handle Hub pushing
                if self.api and self.push_results_to_hub:
                    remasking_dir = extract_remasking(gen_kwargs)
                    repr_param_dir = _resolve_repr_param_value(gen_kwargs)
                    run_id = self.__class__.run_id or "unknown"
                    repo_id = (
                        self.results_repo
                        if self.public_repo
                        else self.results_repo_private
                    )
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(file_path),
                        path_in_repo=os.path.join(
                            self.general_config_tracker.model_name,
                            remasking_dir,
                            repr_param_dir,
                            run_id,
                            f"results_{task_name}.json",
                        ),
                        repo_type="dataset",
                        commit_message=f"Adding results for {self.general_config_tracker.model_name}",
                    )

            except Exception as e:
                logger.warning("Could not save aggregated results: %s", e)

    def save_results_samples(
        self,
        task_name: str,
        samples: dict,
    ) -> None:
        """Save per-sample results with structured subdirectory.

        Reuses the output directory determined by save_results_aggregated
        (which is always called first by lm-eval).
        """
        if self.output_path:
            try:
                logger.info("Saving per-sample results for: %s", task_name)

                # Reuse the path from save_results_aggregated
                path = self._gen_kwargs_output_dir
                if path is None:
                    # Fallback: no gen_kwargs info, use model dir directly
                    path = Path(self.output_path)
                    path = path / self.general_config_tracker.model_name_sanitized
                    path.mkdir(parents=True, exist_ok=True)

                file_path = path / f"samples_{task_name}.jsonl"

                for sample in samples:
                    arguments = {}
                    for i, arg in enumerate(sample["arguments"]):
                        arguments[f"gen_args_{i}"] = {}
                        for j, tmp in enumerate(arg):
                            arguments[f"gen_args_{i}"][f"arg_{j}"] = tmp

                    sample["resps"] = sanitize_list(sample["resps"])
                    sample["filtered_resps"] = sanitize_list(sample["filtered_resps"])
                    sample["arguments"] = arguments
                    sample["target"] = str(sample["target"])

                    sample_dump = (
                        json.dumps(
                            sample,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(sample_dump)

                logger.info("Samples saved to %s", file_path)

            except Exception as e:
                logger.warning("Could not save sample results for %s: %s", task_name, e)
