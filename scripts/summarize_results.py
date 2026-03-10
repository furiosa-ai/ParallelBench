#!/usr/bin/env python3
"""Summarize lm-eval results into a single CSV file.

Recursively scans a results directory for results_*.json files and extracts
metrics, gen_kwargs, and task metadata into a flat CSV.

Usage:
    python scripts/summarize_results.py results/
    python scripts/summarize_results.py results/ --output summary.csv
    python scripts/summarize_results.py results/ --output summary.csv --sort task,steps,block_length
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Metric keys to extract from results (in display order)
METRIC_KEYS = [
    "score",
    "score_strict",
    "nfe",
    "input_length",
    "output_length",
    "dec_order_kendall",
    "dec_order_spearman",
    "dec_order_kendall_ignore_pad",
    "dec_order_spearman_ignore_pad",
]

# Generation kwargs keys to extract (in display order)
GEN_KWARGS_KEYS = [
    "steps",
    "block_length",
    "remasking",
    "max_tokens",
    "temperature",
    "alg_temp",
    "alg_threshold",
    "alg_factor",
]

# CSV column order
COLUMNS = [
    "model",
    "task",
    *GEN_KWARGS_KEYS,
    *METRIC_KEYS,
    "n_samples",
    "results_file",
]


def extract_rows_from_results(results_file: Path) -> list[dict]:
    """Extract one row per task from a results JSON file."""
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model_name", data.get("config", {}).get("model", "unknown"))
    config = data.get("config", {})
    cli_gen_kwargs = config.get("gen_kwargs") or {}
    task_results = data.get("results", {})
    task_configs = data.get("configs", {})
    n_samples = data.get("n-samples", {})

    rows = []
    for task_name, metrics in task_results.items():
        task_config = task_configs.get(task_name, {})
        task_generation_kwargs = task_config.get("generation_kwargs", {})

        # Merge: task YAML generation_kwargs as base, CLI gen_kwargs as override
        merged_gen_kwargs = {**task_generation_kwargs, **cli_gen_kwargs}

        row = {
            "model": model,
            "task": task_name,
            "results_file": str(results_file),
        }

        # Extract gen_kwargs
        for key in GEN_KWARGS_KEYS:
            row[key] = merged_gen_kwargs.get(key, "")

        # Extract metrics (strip the ',none' suffix from lm-eval metric keys)
        for metric in METRIC_KEYS:
            value = metrics.get(f"{metric},none", "")
            if value == "N/A":
                value = ""
            row[metric] = value

        # Extract effective sample count
        task_n_samples = n_samples.get(task_name, {})
        row["n_samples"] = task_n_samples.get("effective", "")

        rows.append(row)

    return rows


def summarize_results(
    results_dir: Path,
    sort_keys: list[str] | None = None,
) -> list[dict]:
    """Scan results directory and collect all rows."""
    results_files = sorted(results_dir.rglob("results_*.json"))

    if not results_files:
        print(f"No results_*.json files found in {results_dir}", file=sys.stderr)
        return []

    all_rows = []
    for results_file in results_files:
        try:
            rows = extract_rows_from_results(results_file)
            all_rows.extend(rows)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {results_file}: {e}", file=sys.stderr)

    if sort_keys:

        def sort_key(row):
            values = []
            for key in sort_keys:
                val = row.get(key, "")
                # Try numeric sort for numeric values
                try:
                    values.append((0, float(val)))
                except (ValueError, TypeError):
                    values.append((1, str(val)))
            return values

        all_rows.sort(key=sort_key)

    return all_rows


def write_csv(rows: list[dict], output: Path | None) -> None:
    """Write rows to CSV file or stdout."""
    if not rows:
        return

    if output:
        f = open(output, "w", newline="", encoding="utf-8")
    else:
        f = sys.stdout

    try:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            f.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize lm-eval results into CSV")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root directory containing results_*.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file path (default: stdout)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="task,block_length,steps",
        help="Comma-separated column names to sort by (default: task,block_length,steps)",
    )
    args = parser.parse_args()

    sort_keys = [k.strip() for k in args.sort.split(",")] if args.sort else None
    rows = summarize_results(args.results_dir, sort_keys=sort_keys)
    write_csv(rows, args.output)

    if args.output and rows:
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
