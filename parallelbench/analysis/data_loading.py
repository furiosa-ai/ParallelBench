"""Shared data loading utilities for ParallelBench analysis.

Provides functions and constants for scanning result directories,
parsing result JSON files, and collecting rows for analysis.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from parallelbench.models.unmasking_registry import get_all_config_params

logger = logging.getLogger(__name__)

# Matches run directories named with timestamp prefix: YYYYMMDD_HHMMSS
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}")

METRIC_KEYS = [
    "score",
    "score_strict",
    "nfe",
    "tokens_per_step",
    "input_length",
    "output_length",
]

_BASE_GENERATION_KWARGS_KEYS = [
    "k",
    "steps",
    "block_length",
    "unmasking",
    "max_tokens",
    "temperature",
    "alg_temp",
]

# Dynamically include all config params from the unmasking registry
GENERATION_KWARGS_KEYS = [
    *_BASE_GENERATION_KWARGS_KEYS,
    *sorted(get_all_config_params() - set(_BASE_GENERATION_KWARGS_KEYS)),
]


def extract_rows_from_results(results_file: Path) -> list[dict]:
    """Extract one row per task from a results JSON file."""
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model_name", data.get("config", {}).get("model", "unknown"))
    config = data.get("config", {})
    cli_generation_kwargs = config.get("gen_kwargs") or {}
    task_results = data.get("results", {})
    task_configs = data.get("configs", {})
    n_samples = data.get("n-samples", {})

    rows = []
    for task_name, metrics in task_results.items():
        task_config = task_configs.get(task_name, {})
        task_generation_kwargs = task_config.get("generation_kwargs", {})
        merged_generation_kwargs = {**task_generation_kwargs, **cli_generation_kwargs}

        row = {
            "model": model,
            "task": task_name,
            "results_file": str(results_file),
        }

        for key in GENERATION_KWARGS_KEYS:
            row[key] = merged_generation_kwargs.get(key, "")

        for metric in METRIC_KEYS:
            value = metrics.get(f"{metric},none", "")
            if value == "N/A":
                value = ""
            row[metric] = value

        # Fallback: compute tokens_per_step from gen_kwargs if not in metrics
        if not row.get("tokens_per_step"):
            try:
                nfe = float(row["nfe"])
                max_tokens = int(row["max_tokens"])
                row["tokens_per_step"] = max_tokens / nfe if nfe > 0 else ""
            except (ValueError, TypeError):
                row["tokens_per_step"] = ""

        # Compute k = max_tokens / steps (tokens unmasked per step)
        if not row.get("k"):
            try:
                max_tokens_val = int(row["max_tokens"])
                steps_val = int(row["steps"])
                row["k"] = max_tokens_val / steps_val if steps_val > 0 else ""
            except (ValueError, TypeError):
                row["k"] = ""

        task_n_samples = n_samples.get(task_name, {})
        row["n_samples"] = task_n_samples.get("effective", "")

        rows.append(row)

    return rows


def find_latest_result_files(results_dir: Path) -> list[Path]:
    """Find the latest result file per (repr_param group, task) combination.

    This is a file-level selection: when category-specific scripts produce
    results in different run directories under the same repr_param group,
    each task's latest file is selected independently.

    Algorithm:
    1. Glob all results_*.json files under results_dir
    2. Group files by (grandparent path, filename) — i.e., (repr_param, task)
    3. Within each group, filter to files whose parent dir matches TIMESTAMP_DIR_RE
    4. If any timestamp dirs exist, pick the file from the lexicographically last one
    5. If NO timestamp dirs exist, fall back to the lexicographically last parent dir
    6. Return the list of selected result file paths
    """
    all_results_files = list(results_dir.rglob("results_*.json"))
    if not all_results_files:
        return []

    # Group by (repr_param dir, filename) so each task is resolved independently
    groups: dict[tuple[Path, str], list[Path]] = {}
    for results_file in all_results_files:
        run_dir = results_file.parent
        group_key = (run_dir.parent, results_file.name)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(results_file)

    selected_files: list[Path] = []
    for files in groups.values():
        timestamp_files = [f for f in files if TIMESTAMP_DIR_RE.match(f.parent.name)]
        if timestamp_files:
            selected_files.append(max(timestamp_files, key=lambda f: f.parent.name))
        else:
            selected_files.append(max(files, key=lambda f: f.parent.name))

    return selected_files


def collect_rows(results_dir: Path, sort_keys: list[str] | None = None) -> list[dict]:
    """Scan results directory and collect all rows.

    Matches both legacy timestamp filenames (results_2026-03-10T05-48-12.json)
    and new task-name filenames (results_parallelbench_waiting_line_copy.json).
    """
    results_files = sorted(results_dir.rglob("results_*.json"))

    if not results_files:
        logger.warning("No results found in %s", results_dir)
        return []

    all_rows = []
    for results_file in results_files:
        try:
            rows = extract_rows_from_results(results_file)
            all_rows.extend(rows)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("skipping %s: %s", results_file, e)

    if sort_keys:

        def sort_key(row):
            values = []
            for key in sort_keys:
                val = row.get(key, "")
                try:
                    values.append((0, float(val)))
                except (ValueError, TypeError):
                    values.append((1, str(val)))
            return values

        all_rows.sort(key=sort_key)

    return all_rows
