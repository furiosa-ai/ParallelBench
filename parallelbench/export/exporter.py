"""Core export logic for ParallelBench GitHub Pages data files.

Reads ParallelBench evaluation results and generates:
- leaderboard/{model_id}.json  — PBx scores per strategy
- figures/{model_id}/{strategy_id}.csv — (task, tps, accuracy) data points
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from parallelbench.analysis.data_loading import (
    extract_rows_from_results,
    find_latest_result_files,
    collect_rows,
)
from parallelbench.analysis.pb_score import compute_pb_scores
from parallelbench.models.unmasking_registry import get_method_type
from parallelbench.export.mapping import (
    get_model_id,
    get_strategy_id,
    get_task_id,
)

logger = logging.getLogger(__name__)

_EXPORT_THRESHOLDS = [80, 75, 70]


_KNOWN_AGGREGATE_TASKS = {
    "parallelbench_all",
    "parallelbench_puzzles",
    "parallelbench_text_writing",
    "parallelbench_waiting_line",
}


def _filter_leaf_tasks(rows: list[dict]) -> list[dict]:
    """Filter out aggregate task rows, keeping only leaf tasks.

    A task is considered aggregate if it is in the known aggregate set,
    or if another task name starts with it followed by '_'.
    """
    all_tasks = {row.get("task", "") for row in rows}
    aggregate_tasks = _KNOWN_AGGREGATE_TASKS | {
        t
        for t in all_tasks
        if any(other != t and other.startswith(t + "_") for other in all_tasks)
    }
    return [row for row in rows if row.get("task", "") not in aggregate_tasks]


def generate_leaderboard_json(
    rows: list[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Generate leaderboard JSON files per model.

    For each model, computes PBx scores per strategy and writes to
    {output_dir}/leaderboard/{model_id}.json.

    Output format per model:
        {
            "thresholds": [80, 75, 70],
            "results": {
                "strategy-id": {"80": score, "75": score, "70": score},
                ...
            }
        }

    Args:
        rows: Result rows (all tasks, will be filtered to leaf tasks).
        output_dir: Root output directory.
        dry_run: If True, skip writing files and only return the summary.

    Returns:
        Summary dict mapping model_id to list of strategy_ids written.
    """
    leaf_rows = _filter_leaf_tasks(rows)

    # Group rows by model
    model_groups: dict[str, list[dict]] = {}
    for row in leaf_rows:
        model = row.get("model", "")
        model_groups.setdefault(model, []).append(row)

    summary: dict[str, list[str]] = {}

    for model, model_rows in sorted(model_groups.items()):
        model_id = get_model_id(model)
        if model_id is None:
            logger.warning("Unmapped model '%s' — skipping leaderboard export", model)
            continue

        # Group by unmasking strategy within this model
        strategy_groups: dict[str, list[dict]] = {}
        for row in model_rows:
            unmasking = row.get("unmasking", "")
            strategy_groups.setdefault(unmasking, []).append(row)

        results: dict[str, dict[str, float]] = {}
        for unmasking, strategy_rows in sorted(strategy_groups.items()):
            strategy_id = get_strategy_id(unmasking)
            if strategy_id is None:
                logger.warning(
                    "Unmapped strategy '%s' for model '%s' — skipping",
                    unmasking,
                    model,
                )
                continue

            pb_scores = compute_pb_scores(strategy_rows, thresholds=_EXPORT_THRESHOLDS)

            strategy_result: dict[str, float] = {}
            for threshold in _EXPORT_THRESHOLDS:
                key = f"PB{threshold}"
                value = pb_scores.get(key)
                strategy_result[str(threshold)] = (
                    float(value) if value is not None else 0.0
                )

            results[strategy_id] = strategy_result

        payload = {
            "thresholds": _EXPORT_THRESHOLDS,
            "results": results,
        }

        if not dry_run:
            leaderboard_dir = output_dir / "leaderboard"
            leaderboard_dir.mkdir(parents=True, exist_ok=True)
            output_file = leaderboard_dir / f"{model_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Wrote %s", output_file)

        summary[model_id] = list(results.keys())

    return summary


def generate_figures_csv(
    rows: list[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Generate figures CSV files per (model, strategy).

    For each (model, unmasking) group, writes a CSV with columns:
        task, tps, accuracy

    Rows are sorted by (task ascending, tps ascending). After all per-task
    rows, 'avg' rows are appended — one per unique TPS value, with mean
    accuracy across all tasks at that TPS.

    Written to: {output_dir}/figures/{model_id}/{strategy_id}.csv

    Args:
        rows: Result rows (all tasks, will be filtered to leaf tasks).
        output_dir: Root output directory.
        dry_run: If True, skip writing files and only return the summary.

    Returns:
        Summary dict mapping "{model_id}/{strategy_id}" to row count written.
    """
    leaf_rows = _filter_leaf_tasks(rows)

    # Group rows by (model, unmasking)
    group_map: dict[tuple[str, str], list[dict]] = {}
    for row in leaf_rows:
        model = row.get("model", "")
        unmasking = row.get("unmasking", "")
        group_map.setdefault((model, unmasking), []).append(row)

    summary: dict[str, int] = {}

    for (model, unmasking), group_rows in sorted(group_map.items()):
        model_id = get_model_id(model)
        if model_id is None:
            logger.warning("Unmapped model '%s' — skipping figures export", model)
            continue

        strategy_id = get_strategy_id(unmasking)
        if strategy_id is None:
            logger.warning(
                "Unmapped strategy '%s' for model '%s' — skipping figures export",
                unmasking,
                model,
            )
            continue

        # Determine method type for TPS calculation
        try:
            method_type = get_method_type(unmasking)
        except KeyError:
            method_type = "unknown"

        # Build per-task data rows
        csv_rows: list[dict] = []
        for row in group_rows:
            task_name = row.get("task", "")
            task_id = get_task_id(task_name)
            if task_id is None:
                logger.warning(
                    "Unmapped task '%s' — skipping row in figures export", task_name
                )
                continue

            try:
                accuracy = float(row["score"])
            except (KeyError, ValueError, TypeError):
                continue

            # TPS: for topk methods use k; for others use measured tokens_per_step
            tps: float | None = None
            if method_type == "topk":
                try:
                    tps = float(row["k"])
                except (KeyError, ValueError, TypeError):
                    try:
                        max_tokens = int(row["max_tokens"])
                        steps = int(row["steps"])
                        tps = max_tokens / steps if steps > 0 else None
                    except (KeyError, ValueError, TypeError):
                        pass
            else:
                try:
                    tps = float(row["tokens_per_step"])
                except (KeyError, ValueError, TypeError):
                    try:
                        nfe = float(row["nfe"])
                        max_tokens = int(row["max_tokens"])
                        tps = max_tokens / nfe if nfe > 0 else None
                    except (KeyError, ValueError, TypeError):
                        pass

            if tps is None:
                continue

            csv_rows.append({"task": task_id, "tps": tps, "accuracy": accuracy})

        if not csv_rows:
            continue

        # Sort by (task ascending, tps ascending)
        csv_rows.sort(key=lambda r: (r["task"], r["tps"]))

        # Compute avg rows: mean accuracy per unique TPS across all tasks
        tps_to_accuracies: dict[float, list[float]] = {}
        for r in csv_rows:
            tps_to_accuracies.setdefault(r["tps"], []).append(r["accuracy"])

        avg_rows: list[dict] = []
        for tps_val in sorted(tps_to_accuracies.keys()):
            accuracies = tps_to_accuracies[tps_val]
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_rows.append({"task": "avg", "tps": tps_val, "accuracy": avg_accuracy})

        all_rows = csv_rows + avg_rows
        total_rows = len(all_rows)

        if not dry_run:
            figures_dir = output_dir / "figures" / model_id
            figures_dir.mkdir(parents=True, exist_ok=True)
            output_file = figures_dir / f"{strategy_id}.csv"
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["task", "tps", "accuracy"])
                for r in all_rows:
                    writer.writerow([r["task"], r["tps"], r["accuracy"]])
            logger.info("Wrote %s (%d rows)", output_file, total_rows)

        summary[f"{model_id}/{strategy_id}"] = total_rows

    return summary


def export_all(
    results_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    all_runs: bool = False,
) -> dict:
    """Load results and generate all GitHub Pages export files.

    Loads rows from results_dir using find_latest_result_files() (or
    collect_rows() when all_runs=True), then calls generate_leaderboard_json()
    and generate_figures_csv().

    Args:
        results_dir: Root directory containing results_*.json files.
        output_dir: Root output directory for exported files.
        dry_run: If True, skip writing files and only return the summary.
        all_runs: If True, use all runs instead of latest only.

    Returns:
        Combined summary dict with keys 'leaderboard' and 'figures'.
    """
    if all_runs:
        rows = collect_rows(results_dir)
    else:
        latest_files = find_latest_result_files(results_dir)
        rows = []
        for results_file in latest_files:
            try:
                rows.extend(extract_rows_from_results(results_file))
            except (ValueError, KeyError) as e:
                logger.warning("Skipping %s: %s", results_file, e)

    if not rows:
        logger.warning("No result rows found in %s", results_dir)
        return {"leaderboard": {}, "figures": {}}

    # Warn about unmapped models and strategies
    seen_models = {row.get("model", "") for row in rows}
    seen_strategies = {row.get("unmasking", "") for row in rows}

    for model in sorted(seen_models):
        if model and get_model_id(model) is None:
            logger.warning("No GitHub Pages mapping for model '%s'", model)

    for strategy in sorted(seen_strategies):
        if strategy and get_strategy_id(strategy) is None:
            logger.warning("No GitHub Pages mapping for strategy '%s'", strategy)

    leaderboard_summary = generate_leaderboard_json(rows, output_dir, dry_run=dry_run)
    figures_summary = generate_figures_csv(rows, output_dir, dry_run=dry_run)

    return {
        "leaderboard": leaderboard_summary,
        "figures": figures_summary,
    }
