"""ParallelBench CLI.

Usage:
    pb eval [lm-eval options]     Run lm-eval evaluation
    pb data [options]             Generate benchmark data
    pb browse [task] [options]    Browse benchmark tasks and samples
"""

import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

COMMANDS = {
    "eval": (
        "Run lm-eval evaluation",
        "pb eval --model parallelbench_llada --model_args ...",
    ),
    "data": ("Generate benchmark data", "pb data --output_dir ./output"),
    "browse": ("Browse benchmark tasks and samples", "pb browse waiting_line/copy"),
}


def _print_help():
    """Print Rich-formatted help with commands table."""
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Command", style="bold green")
    table.add_column("Description")
    table.add_column("Example", style="dim")

    for cmd, (desc, example) in COMMANDS.items():
        table.add_row(cmd, desc, example)

    help_text = Text.assemble(
        ("Run ", ""),
        ("pb <command> --help", "bold"),
        (" for more information.", ""),
    )

    panel = Panel(
        table,
        title="[bold]ParallelBench CLI[/bold]",
        subtitle=help_text,
        border_style="blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def main():
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "eval":
        from parallelbench.cli.eval import main as eval_main

        eval_main()
    elif command == "data":
        from parallelbench.cli.data import main as data_main

        data_main()
    elif command == "browse":
        from parallelbench.cli.browse import main as browse_main

        browse_main()
    else:
        console.print(
            f"\n[bold red]Error:[/bold red] Unknown command [yellow]'{command}'[/yellow]"
        )
        console.print(f"Available commands: [green]{', '.join(COMMANDS)}[/green]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
