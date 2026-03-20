"""Analyze ParallelBench evaluation results.

Usage:
    pb analyze leaderboard results/              PBx leaderboard ranked by PB80
    pb analyze best results/                     Best method per model summary
    pb analyze detail results/                   Per-(model, unmasking) detailed tables
    pb analyze detail results/ --export out.csv  Export detailed results to CSV
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

# ---------------------------------------------------------------------------
# Display name mappings
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "GSAI-ML/LLaDA-1.5": "LLaDA 1.5",
    "GSAI-ML/LLaDA-8B-Instruct": "LLaDA 8B Instruct",
    "Dream-org/Dream-v0-Instruct-7B": "Dream 7B",
    "apple/DiffuCoder-7B-Instruct": "DiffuCoder 7B",
    "Gen-Verse/TraDo-4B-Instruct": "TraDo 4B",
    "Gen-Verse/TraDo-8B-Instruct": "TraDo 8B",
}

METHOD_DISPLAY_NAMES = {
    "confidence_topk": "Confidence Top-K",
    "confidence_threshold": "Confidence Threshold",
    "confidence_factor": "Confidence Factor",
    "entropy_topk": "Entropy Top-K",
    "topk_margin": "Top-K Margin",
    "random": "Random",
    "left_to_right": "Left-to-Right",
    "origin": "Origin",
    "klass": "KLASS",
}


# ---------------------------------------------------------------------------
# Data loading (shared across subcommands)
# ---------------------------------------------------------------------------


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


def _find_latest_result_files(results_dir: Path) -> list[Path]:
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


def _load_rows(args: argparse.Namespace) -> list[dict]:
    """Load rows from results directory based on --all-runs flag."""
    results_dir = args.results_dir
    sort_keys = None
    if hasattr(args, "sort") and args.sort:
        sort_keys = [k.strip() for k in args.sort.split(",")]

    if args.all_runs:
        rows = _collect_rows(results_dir, sort_keys=sort_keys)
    else:
        latest_files = _find_latest_result_files(results_dir)
        if latest_files:
            console_err.print(
                f"[dim]Using {len(latest_files)} latest result file(s)[/dim]"
            )
            rows = []
            for results_file in latest_files:
                try:
                    rows.extend(_extract_rows_from_results(results_file))
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

                rows.sort(key=sort_key)
        else:
            rows = _collect_rows(results_dir, sort_keys=sort_keys)

    if not rows:
        console_err.print("[bold red]No results found[/bold red]")
        sys.exit(1)

    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


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


def _format_pbx(value: float | None) -> str:
    """Format a PBx score for Rich table display."""
    if value is None:
        return "[dim]-[/dim]"
    return f"[cyan]{value:.1f}[/cyan]"


def _display_name(model: str) -> str:
    """Return display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model)


def _method_name(method: str) -> str:
    """Return display name for an unmasking method."""
    return METHOD_DISPLAY_NAMES.get(method, method)


# ---------------------------------------------------------------------------
# Detail subcommand helpers (existing logic)
# ---------------------------------------------------------------------------


def _compute_average_rows(rows: list[dict], method_type: str) -> list[dict]:
    """Compute per-hyperparameter average rows when multiple tasks exist.

    Groups rows by the representative hyperparameter (k, alg_threshold, or
    alg_factor) and averages numeric metrics across tasks within each group.
    Returns an empty list when fewer than 2 distinct tasks exist.

    Rows without ``nfe`` are excluded to match PBx score computation, which
    skips group-level aggregate rows (e.g., "puzzles", "text_writing") that
    lack per-sample generation metrics.
    """
    # Exclude group-level aggregate rows whose task name is a strict prefix
    # of another task (e.g., "puzzles" is a prefix of "puzzles_sudoku_n4").
    # This matches PBx score computation which only uses leaf-level tasks.
    all_tasks = {row.get("task") for row in rows}
    group_tasks = {
        t for t in all_tasks if any(o != t and o.startswith(t + "_") for o in all_tasks)
    }
    leaf_rows = [r for r in rows if r.get("task") not in group_tasks]
    tasks = {row.get("task") for row in leaf_rows}
    if len(tasks) < 2:
        return []

    group_key_map = {
        "topk": "k",
        "threshold": "alg_threshold",
        "factor": "alg_factor",
    }
    group_key = group_key_map.get(method_type)
    if not group_key:
        return []

    groups: dict[str, list[dict]] = {}
    for row in leaf_rows:
        key = str(row.get(group_key, ""))
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    avg_rows = []
    for param_value, group_rows in sorted(groups.items()):
        avg_row: dict = {"task": "Average", group_key: param_value}
        for metric in METRIC_KEYS:
            values = []
            for r in group_rows:
                v = r.get(metric)
                if v not in ("", None):
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
            avg_row[metric] = sum(values) / len(values) if values else ""
        avg_rows.append(avg_row)

    # Sort by tokens_per_step ascending to match the main table sort order
    def _tps_sort_key(row):
        try:
            return float(row.get("tokens_per_step", 0))
        except (ValueError, TypeError):
            return 0

    avg_rows.sort(key=_tps_sort_key)

    return avg_rows


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
    rows: list[dict],
    title: str | None = None,
    columns: list[str] | None = None,
    average_rows: list[dict] | None = None,
) -> None:
    """Print a Rich-formatted results table.

    When *average_rows* is provided, they are appended after a section
    separator with bold styling to distinguish them from individual results.
    """
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

    if average_rows:
        table.add_section()
        for row in average_rows:
            styled = [
                f"[bold]{_format_value(col, row.get(col, ''))}[/bold]"
                for col in display_columns
            ]
            table.add_row(*styled)

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


# ---------------------------------------------------------------------------
# Subcommand: leaderboard
# ---------------------------------------------------------------------------


def _build_leaderboard_records(rows: list[dict]) -> list[dict]:
    """Build PBx leaderboard records from result rows.

    Returns a list of dicts with: model, method, PB90, PB80, PB70, PB60.
    Sorted by PB80 descending.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = _get_group_key(row)
        groups.setdefault(key, []).append(row)

    records = []
    for (model, unmasking), group_rows in sorted(groups.items()):
        pb_scores = compute_pb_scores(group_rows, thresholds=DEFAULT_THRESHOLDS)
        record = {"model": model, "method": unmasking}
        record.update(pb_scores)
        records.append(record)

    records.sort(key=lambda r: r.get("PB80") or -1, reverse=True)
    return records


def _print_leaderboard_table(records: list[dict], title: str) -> None:
    """Print a PBx leaderboard as a Rich table."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Model", min_width=20)
    table.add_column("Method", min_width=18)
    for t in DEFAULT_THRESHOLDS:
        table.add_column(f"PB{t}", justify="right", min_width=6)

    for rank, record in enumerate(records, 1):
        table.add_row(
            str(rank),
            _display_name(record["model"]),
            _method_name(record["method"]),
            *[_format_pbx(record.get(f"PB{t}")) for t in DEFAULT_THRESHOLDS],
        )

    console.print()
    console.print(table)
    console.print()


def _cmd_leaderboard(args: argparse.Namespace) -> None:
    """Show PBx leaderboard ranked by PB80."""
    rows = _load_rows(args)
    records = _build_leaderboard_records(rows)
    _print_leaderboard_table(
        records,
        title="PBx Leaderboard (max TPS achieving >= x% avg score)",
    )


# ---------------------------------------------------------------------------
# Subcommand: best
# ---------------------------------------------------------------------------


def _cmd_best(args: argparse.Namespace) -> None:
    """Show best method per model."""
    rows = _load_rows(args)
    records = _build_leaderboard_records(rows)

    # Keep only the best method per model (by PB80)
    seen_models: set[str] = set()
    best_records = []
    for record in records:
        if record["model"] not in seen_models:
            seen_models.add(record["model"])
            best_records.append(record)

    _print_leaderboard_table(
        best_records,
        title="PBx Best Method per Model (ranked by PB80)",
    )


# ---------------------------------------------------------------------------
# Subcommand: detail
# ---------------------------------------------------------------------------


def _cmd_detail(args: argparse.Namespace) -> None:
    """Show detailed per-(model, unmasking) results tables."""
    rows = _load_rows(args)

    if hasattr(args, "export") and args.export:
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
        avg_rows = _compute_average_rows(group_rows, method_type)
        _print_results_table(
            group_rows,
            title=f"{model} / {unmasking}",
            columns=columns,
            average_rows=avg_rows,
        )
        _print_pb_scores(group_rows)

    console.print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root directory containing results_*.json files",
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ParallelBench evaluation results",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # leaderboard
    parser_leaderboard = subparsers.add_parser(
        "leaderboard",
        help="PBx leaderboard ranked by PB80",
    )
    _add_common_args(parser_leaderboard)
    parser_leaderboard.set_defaults(func=_cmd_leaderboard)

    # best
    parser_best = subparsers.add_parser(
        "best",
        help="Best method per model summary",
    )
    _add_common_args(parser_best)
    parser_best.set_defaults(func=_cmd_best)

    # detail
    parser_detail = subparsers.add_parser(
        "detail",
        help="Per-(model, unmasking) detailed tables with PBx scores",
    )
    _add_common_args(parser_detail)
    parser_detail.add_argument(
        "--sort",
        type=str,
        default="task,tokens_per_step,alg_threshold",
        help="Comma-separated column names to sort by (default: task,tokens_per_step,alg_threshold)",
    )
    parser_detail.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export results to CSV file",
    )
    parser_detail.set_defaults(func=_cmd_detail)

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
