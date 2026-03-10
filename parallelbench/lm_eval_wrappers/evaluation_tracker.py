"""Custom EvaluationTracker that organizes results by gen_kwargs subdirectory.

lm-eval's default output structure:
    {output_path}/{model_sanitized}/results_{timestamp}.json

ParallelBench output structure:
    {output_path}/{model_sanitized}/{gen_kwargs_dir}/results_{timestamp}.json

This allows sweep experiments with different gen_kwargs to be cleanly separated
while sharing the same model directory.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from lm_eval.loggers.evaluation_tracker import EvaluationTracker, sanitize_list
from lm_eval.utils import handle_non_serializable, hash_string

logger = logging.getLogger(__name__)


def build_gen_kwargs_dirname(gen_kwargs: dict | None) -> str:
    """Build a directory name from gen_kwargs dict.

    Produces a compact, filesystem-safe string like 'bl32_s32_low_confidence'.
    Keys are abbreviated and ordered for readability:
        block_length -> bl, steps -> s, remasking -> (value only),
        alg_threshold -> at, alg_factor -> af, temperature -> t
    Unknown keys are appended as key=value pairs sorted alphabetically.

    Returns 'default' when gen_kwargs is empty or None.
    """
    if not gen_kwargs:
        return "default"

    abbreviations = {
        "block_length": "bl",
        "steps": "s",
        "max_tokens": "mt",
        "remasking": "",
        "alg_threshold": "at",
        "alg_factor": "af",
        "temperature": "t",
        "alg_temp": "algt",
    }

    # Ordered keys for consistent, readable directory names
    ordered_keys = [
        "block_length",
        "steps",
        "remasking",
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

        if abbr == "":
            # remasking: use value directly
            parts.append(str(value))
        else:
            parts.append(f"{abbr}{value}")

    # Append any unknown keys sorted alphabetically
    for key in sorted(gen_kwargs.keys()):
        if key in used_keys:
            continue
        # Skip lm-eval internal keys
        if key in ("until", "do_sample"):
            continue
        value = gen_kwargs[key]
        if isinstance(value, float) and value == int(value):
            value = int(value)
        parts.append(f"{key}{value}")

    return "_".join(parts) if parts else "default"


class ParallelBenchEvaluationTracker(EvaluationTracker):
    """EvaluationTracker subclass that adds gen_kwargs subdirectory to output path.

    Overrides save_results_aggregated and save_results_samples to insert
    a gen_kwargs-derived subdirectory between the model directory and result files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will be set by save_results_aggregated for reuse in save_results_samples
        self._gen_kwargs_output_dir: Path | None = None

    def _resolve_output_dir(self, gen_kwargs: dict | None) -> Path:
        """Build the output directory path including gen_kwargs subdirectory."""
        path = Path(self.output_path if self.output_path else Path.cwd())
        path = path / self.general_config_tracker.model_name_sanitized
        path = path / build_gen_kwargs_dirname(gen_kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict | None = None,
    ) -> None:
        """Save aggregated results with gen_kwargs subdirectory."""
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
                path = self._resolve_output_dir(gen_kwargs)
                self._gen_kwargs_output_dir = path

                self.date_id = datetime.now().isoformat().replace(":", "-")
                file_path = path / f"results_{self.date_id}.json"
                file_path.open("w", encoding="utf-8").write(dumped)

                logger.info(
                    "Results saved to %s (gen_kwargs_dir: %s)",
                    file_path,
                    build_gen_kwargs_dirname(gen_kwargs),
                )

                # Handle Hub pushing
                if self.api and self.push_results_to_hub:
                    gen_kwargs_dir = build_gen_kwargs_dirname(gen_kwargs)
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
                            gen_kwargs_dir,
                            f"results_{self.date_id}.json",
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
        """Save per-sample results with gen_kwargs subdirectory.

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

                file_path = path / f"samples_{task_name}_{self.date_id}.jsonl"

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
