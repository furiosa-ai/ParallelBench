"""Export ParallelBench results to GitHub Pages-compatible data files.

Usage:
    pb export --output <dir>              Generate leaderboard JSON + figures CSV
    pb export --output <dir> --dry-run    Preview without writing files
    pb export --results-dir <dir>         Override results directory
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
console_err = Console(stderr=True)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ParallelBench results to GitHub Pages-compatible data files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Root directory containing results_*.json files (default: ./results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview what would be written without writing any files",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        default=False,
        help=(
            "Use all runs instead of the latest only. "
            "By default, the latest run per repr_param group is selected."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    results_dir: Path = args.results_dir
    output_dir: Path = args.output
    dry_run: bool = args.dry_run
    all_runs: bool = args.all_runs

    if not results_dir.exists():
        console_err.print(
            f"[bold red]Error:[/bold red] Results directory not found: {results_dir}"
        )
        sys.exit(1)

    if dry_run:
        console_err.print("[dim]Dry run — no files will be written[/dim]")

    from parallelbench.export.exporter import export_all

    summary = export_all(
        results_dir=results_dir,
        output_dir=output_dir,
        dry_run=dry_run,
        all_runs=all_runs,
    )

    leaderboard = summary.get("leaderboard", {})
    figures = summary.get("figures", {})

    if not leaderboard and not figures:
        console_err.print("[bold red]No data exported.[/bold red]")
        sys.exit(1)

    # Print leaderboard summary
    if leaderboard:
        table = Table(
            title="Leaderboard Export",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Model ID", style="bold green")
        table.add_column("Strategies", justify="right")
        for model_id, strategies in sorted(leaderboard.items()):
            table.add_row(model_id, str(len(strategies)))
        console.print()
        console.print(table)

    # Print figures summary
    if figures:
        table = Table(
            title="Figures Export",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Model / Strategy", style="bold green")
        table.add_column("Rows", justify="right")
        for key, row_count in sorted(figures.items()):
            table.add_row(key, str(row_count))
        console.print()
        console.print(table)

    if dry_run:
        console.print("\n[dim]Dry run complete — no files written.[/dim]")
    else:
        console.print(
            f"\n[green]Export complete.[/green] "
            f"Leaderboard: {len(leaderboard)} model(s), "
            f"Figures: {len(figures)} (model, strategy) pair(s)."
        )


if __name__ == "__main__":
    main()
