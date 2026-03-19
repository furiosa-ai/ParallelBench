"""Analyze ParallelBench evaluation results.

Usage:
    pb analyze results/                    Print summary table grouped by (model, unmasking)
    pb analyze results/ --export summary.csv     Export to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from parallelbench.analysis.pb_score import DEFAULT_THRESHOLDS, compute_pb_scores
from parallelbench.models.unmasking_registry import get_method_type

console = Console()
console_err = Console(stderr=True)

# Matches run directories named with timestamp prefix: YYYYMMDD_HHMMSS
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}")

METRIC_KEYS = [
    "score",
    "score_strict",
    "nfe",
    "tokens_per_step",
    "input_length",
    "output_length",
    "dec_order_kendall",
    "dec_order_spearman",
    "dec_order_kendall_ignore_pad",
    "dec_order_spearman_ignore_pad",
]

GENERATION_KWARGS_KEYS = [
    "k",
    "steps",
    "block_length",
    "unmasking",
    "max_tokens",
    "temperature",
    "alg_temp",
    "alg_threshold",
    "alg_factor",
]

DISPLAY_COLUMNS = [
    "task",
    "unmasking",
    "tokens_per_step",
    "nfe",
    "score",
]

CSV_COLUMNS = [
    "model",
    "task",
    *GENERATION_KWARGS_KEYS,
    *METRIC_KEYS,
    "n_samples",
    "results_file",
]


def _extract_rows_from_results(results_file: Path) -> list[dict]:
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


def _find_latest_run_dirs(results_dir: Path) -> list[Path]:
    """Find the latest run directory under each repr_param group.

    Algorithm:
    1. Glob all results_*.json files under results_dir
    2. Group files by grandparent path (the repr_param level)
    3. Within each group, collect unique parent directory names (the run dirs)
    4. Filter to only dirs matching TIMESTAMP_DIR_RE (^\\d{8}_\\d{6})
    5. If any timestamp dirs exist, pick the lexicographically last one
    6. If NO timestamp dirs exist (all legacy dirs), fall back to
       picking the lexicographically last of ALL dirs
    7. Return the list of selected run directory paths
    """
    all_results_files = list(results_dir.rglob("results_*.json"))
    if not all_results_files:
        return []

    # Group run dirs by their parent (repr_param level = grandparent of each results file)
    groups: dict[Path, set[Path]] = {}
    for results_file in all_results_files:
        run_dir = results_file.parent
        group_key = run_dir.parent
        if group_key not in groups:
            groups[group_key] = set()
        groups[group_key].add(run_dir)

    selected_run_dirs: list[Path] = []
    for run_dirs in groups.values():
        timestamp_dirs = [d for d in run_dirs if TIMESTAMP_DIR_RE.match(d.name)]
        if timestamp_dirs:
            selected_run_dirs.append(max(timestamp_dirs, key=lambda d: d.name))
        else:
            # Fall back to lexicographically last of all dirs (e.g. legacy UUID dirs)
            selected_run_dirs.append(max(run_dirs, key=lambda d: d.name))

    return selected_run_dirs


def _collect_rows(results_dir: Path, sort_keys: list[str] | None = None) -> list[dict]:
    """Scan results directory and collect all rows.

    Matches both legacy timestamp filenames (results_2026-03-10T05-48-12.json)
    and new task-name filenames (results_parallelbench_waiting_line_copy.json).
    """
    results_files = sorted(results_dir.rglob("results_*.json"))

    if not results_files:
        console_err.print(
            f"[bold red]No results found[/bold red] in {results_dir}",
        )
        return []

    all_rows = []
    for results_file in results_files:
        try:
            rows = _extract_rows_from_results(results_file)
            all_rows.extend(rows)
        except (json.JSONDecodeError, KeyError) as e:
            console.print(
                f"[yellow]Warning:[/yellow] skipping {results_file}: {e}",
            )

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


def _format_value(key: str, value) -> str:
    """Format a value for Rich table display."""
    if value == "" or value is None:
        return "[dim]-[/dim]"
    if key == "score" or key == "score_strict":
        try:
            v = float(value)
            display = f"{v:.0f}"
            if v >= 90:
                return f"[bold green]{display}[/bold green]"
            elif v >= 70:
                return f"[yellow]{display}[/yellow]"
            else:
                return f"[red]{display}[/red]"
        except (ValueError, TypeError):
            return str(value)
    if key == "nfe":
        try:
            return f"{float(value):.0f}"
        except (ValueError, TypeError):
            return str(value)
    if key == "tokens_per_step":
        try:
            return f"{float(value):.1f}"
        except (ValueError, TypeError):
            return str(value)
    if key == "k":
        try:
            v = float(value)
            return f"{int(v)}" if v == int(v) else f"{v:.1f}"
        except (ValueError, TypeError):
            return str(value)
    if key in ("alg_threshold", "alg_factor"):
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _get_display_columns(rows: list[dict]) -> list[str]:
    """Return display columns, including model column when multiple models exist."""
    models = {row.get("model", "unknown") for row in rows}
    if len(models) > 1:
        return ["model", *DISPLAY_COLUMNS]
    return list(DISPLAY_COLUMNS)


def _get_group_key(row: dict) -> tuple[str, str]:
    """Return (model, unmasking) group key for a row."""
    return (row.get("model", "unknown"), row.get("unmasking", "unknown"))


def _get_columns_for_method_type(method_type: str) -> list[str]:
    """Return display columns based on unmasking method type."""
    if method_type == "topk":
        return ["task", "k", "score"]
    elif method_type == "threshold":
        return ["task", "alg_threshold", "tokens_per_step", "score"]
    elif method_type == "factor":
        return ["task", "alg_factor", "tokens_per_step", "score"]
    return ["task", "tokens_per_step", "score"]


def _print_results_table(
    rows: list[dict], title: str | None = None, columns: list[str] | None = None
) -> None:
    """Print a Rich-formatted results table."""
    if title is None:
        models = sorted({row.get("model", "unknown") for row in rows})
        title = f"Results ({', '.join(models)})"

    display_columns = columns if columns is not None else _get_display_columns(rows)

    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )

    numeric_columns = {
        "score",
        "nfe",
        "tokens_per_step",
        "k",
        "alg_threshold",
        "alg_factor",
    }
    col_config = {
        "model": {"max_width": 30},
        "task": {"max_width": 30},
        "unmasking": {"max_width": 16},
        "tokens_per_step": {"min_width": 5},
        "nfe": {"min_width": 5},
        "score": {"min_width": 6},
        "k": {"min_width": 4},
        "alg_threshold": {"min_width": 9},
        "alg_factor": {"min_width": 9},
    }
    for col in display_columns:
        justify = "right" if col in numeric_columns else "left"
        cfg = col_config.get(col, {})
        table.add_column(
            col,
            justify=justify,
            no_wrap=True,
            overflow="ellipsis",
            **cfg,
        )

    for row in rows:
        display_row = dict(row)
        # Shorten task name for display
        task = display_row.get("task", "")
        if task.startswith("parallelbench_"):
            display_row["task"] = task[len("parallelbench_") :]
        table.add_row(
            *[_format_value(col, display_row.get(col, "")) for col in display_columns]
        )

    console.print()
    console.print(table)


def _print_pb_scores(rows: list[dict]) -> None:
    """Print PBx scores summary."""
    pb_scores = compute_pb_scores(rows, thresholds=DEFAULT_THRESHOLDS)

    if all(v is None for v in pb_scores.values()):
        return

    parts = []
    for name, tps in pb_scores.items():
        if tps is not None:
            parts.append(f"[bold]{name}[/bold]: [cyan]{tps:.1f}[/cyan]")
        else:
            parts.append(f"[bold]{name}[/bold]: [dim]-[/dim]")

    score_line = "  |  ".join(parts)
    panel = Panel(
        score_line,
        title="[bold]PBx Scores[/bold] (max tokens-per-step achieving >= x% average)",
        border_style="green",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def _export_csv(rows: list[dict], output: Path) -> None:
    """Export rows to CSV."""
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"\n[green]Exported {len(rows)} rows to {output}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ParallelBench evaluation results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root directory containing results_*.json files",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="task,block_length,steps",
        help="Comma-separated column names to sort by (default: task,block_length,steps)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        default=False,
        help=(
            "Scan every run directory instead of only the latest. "
            "By default, the latest run per repr_param group is selected "
            "by sorting timestamp-prefixed directories (YYYYMMDD_HHMMSS). "
            "Directories without a timestamp prefix are used as fallback."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    sort_keys = [k.strip() for k in args.sort.split(",")] if args.sort else None

    if args.all_runs:
        rows = _collect_rows(results_dir, sort_keys=sort_keys)
    else:
        latest_dirs = _find_latest_run_dirs(results_dir)
        if latest_dirs:
            console_err.print(f"[dim]Using {len(latest_dirs)} latest run(s)[/dim]")
            rows = []
            for run_dir in latest_dirs:
                rows.extend(_collect_rows(run_dir, sort_keys=sort_keys))
        else:
            rows = _collect_rows(results_dir, sort_keys=sort_keys)

    if not rows:
        sys.exit(1)

    if args.export:
        _export_csv(rows, args.export)
        return

    # Group by (model, unmasking)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = _get_group_key(row)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    for (model, unmasking), group_rows in sorted(groups.items()):
        try:
            method_type = get_method_type(unmasking)
        except KeyError:
            method_type = "unknown"
        columns = _get_columns_for_method_type(method_type)
        _print_results_table(
            group_rows, title=f"{model} / {unmasking}", columns=columns
        )
        _print_pb_scores(group_rows)

    console.print()


if __name__ == "__main__":
    main()
