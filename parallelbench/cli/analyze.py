"""Analyze ParallelBench evaluation results.

Usage:
    pb analyze results/                          Print summary table
    pb analyze results/ --compare remasking      Group by remasking strategy
    pb analyze results/ --export summary.csv     Export to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from parallelbench.analysis.pb_score import DEFAULT_THRESHOLDS, compute_pb_scores

console = Console()
console_err = Console(stderr=True)

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
    "steps",
    "block_length",
    "remasking",
    "max_tokens",
    "temperature",
    "alg_temp",
    "alg_threshold",
    "alg_factor",
]

DISPLAY_COLUMNS = [
    "task",
    "remasking",
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

        task_n_samples = n_samples.get(task_name, {})
        row["n_samples"] = task_n_samples.get("effective", "")

        rows.append(row)

    return rows


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
    return str(value)


def _print_results_table(rows: list[dict], title: str | None = None) -> None:
    """Print a Rich-formatted results table."""
    if title is None:
        models = sorted({row.get("model", "unknown") for row in rows})
        title = f"Results ({', '.join(models)})"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )

    numeric_columns = {"score", "nfe", "tokens_per_step"}
    col_config = {
        "task": {"max_width": 30},
        "remasking": {"max_width": 16},
        "tokens_per_step": {"min_width": 5},
        "nfe": {"min_width": 5},
        "score": {"min_width": 6},
    }
    for col in DISPLAY_COLUMNS:
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
            *[_format_value(col, display_row.get(col, "")) for col in DISPLAY_COLUMNS]
        )

    console.print()
    console.print(table)


def _print_comparison_table(rows: list[dict], group_by: str) -> None:
    """Print results grouped by the specified column."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get(group_by, "unknown"))
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    for group_name, group_rows in sorted(groups.items()):
        _print_results_table(group_rows, title=f"{group_by} = {group_name}")


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
        "--compare",
        type=str,
        default=None,
        choices=["remasking", "model", "task", "steps", "block_length"],
        help="Group results by this column for comparison",
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
        "--latest",
        action="store_true",
        default=True,
        help="Use .latest symlink to find the most recent run (default: True)",
    )
    parser.add_argument(
        "--no-latest",
        action="store_false",
        dest="latest",
        help="Scan all runs instead of just the latest",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    sort_keys = [k.strip() for k in args.sort.split(",")] if args.sort else None

    # If .latest symlinks exist, collect results only from latest runs
    if args.latest:
        latest_links = list(results_dir.rglob(".latest"))
        if latest_links:
            console_err.print(
                f"[dim]Using {len(latest_links)} latest run(s)[/dim]",
            )
            rows = []
            for link in latest_links:
                rows.extend(_collect_rows(link.resolve(), sort_keys=sort_keys))
        else:
            rows = _collect_rows(results_dir, sort_keys=sort_keys)
    else:
        rows = _collect_rows(results_dir, sort_keys=sort_keys)

    if not rows:
        sys.exit(1)

    if args.export:
        _export_csv(rows, args.export)
        return

    if args.compare:
        _print_comparison_table(rows, args.compare)
    else:
        _print_results_table(rows)

    _print_pb_scores(rows)
    console.print()


if __name__ == "__main__":
    main()
