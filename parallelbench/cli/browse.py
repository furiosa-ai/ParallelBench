"""Browse ParallelBench tasks and samples from the command line.

Usage:
    pb browse                              List all tasks grouped by category
    pb browse waiting_line/copy            Show samples from a specific task
    pb browse waiting_line/copy --index 3  Show a specific sample
    pb browse waiting_line/copy --generate Use locally generated data instead of Hub
"""

import argparse
import itertools
import json
import textwrap
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

DEFAULT_HUB_REPO = "furiosa-ai/ParallelBench"

CATEGORY_DISPLAY_NAMES = {
    "waiting_line": "Waiting Line",
    "waiting_line_n15": "Waiting Line (n=15, for API/AR models)",
    "text_writing": "Text Writing",
    "puzzles": "Puzzles",
}

CATEGORY_STYLES = {
    "waiting_line": "cyan",
    "waiting_line_n15": "cyan",
    "text_writing": "green",
    "puzzles": "magenta",
}

TASK_DESCRIPTIONS = {
    "waiting_line/copy": "Copy a list of names exactly as given",
    "waiting_line/insert_index": "Insert a person at a specific position",
    "waiting_line/insert_random": "Insert a person at a random position",
    "waiting_line/remove_index": "Remove a person at a specific position",
    "waiting_line/remove_random": "Remove a random person from the list",
    "waiting_line/replace_index": "Replace a person at a specific position",
    "waiting_line/replace_random": "Replace a random person in the list",
    "waiting_line/reverse": "Reverse the order of the list",
    "waiting_line/shuffle": "Randomly shuffle the list",
    "waiting_line/sort": "Sort the list alphabetically by last name",
    "text_writing/paraphrasing": "Paraphrase a given sentence",
    "text_writing/summarization": "Summarize a given passage",
    "text_writing/words_to_sentence_easy": "Form a sentence from given words (easy)",
    "text_writing/words_to_sentence_medium": "Form a sentence from given words (medium)",
    "text_writing/words_to_sentence_hard": "Form a sentence from given words (hard)",
    "puzzles/latin_square_n4": "Solve a 4x4 Latin square puzzle",
    "puzzles/sudoku_n4": "Solve a 4x4 Sudoku puzzle",
}


def _load_all_task_configs():
    """Load all task configs from YAML files without heavy imports.

    Parses YAML directly to avoid importing parallelbench.datasets (which
    triggers torch/transformers imports and adds ~5s startup time).
    """
    config_dir = (
        Path(__file__).parent.parent / "datasets" / "data" / "task_configs" / "test"
    )
    all_configs = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)

        global_config = raw.get("global_config", {})
        tasks = raw.get("tasks", {})
        category = yaml_file.stem

        for task_short_name, task_cfg in tasks.items():
            task_name = f"{category}/{task_short_name}"
            # Merge global_config defaults (handle per-task overrides in dicts)
            merged_global = {}
            for k, v in global_config.items():
                if isinstance(v, dict):
                    merged_global[k] = v.get(task_short_name, v.get("default"))
                else:
                    merged_global[k] = v
            all_configs[task_name] = {**merged_global, **task_cfg}

    return all_configs


def _print_task_list():
    """Print all tasks grouped by category with sample counts and descriptions."""
    all_configs = _load_all_task_configs()
    task_names = sorted(name for name in all_configs if not name.startswith("_"))

    # Group by category
    categories = {}
    for name in task_names:
        category = name.split("/")[0]
        categories.setdefault(category, []).append(name)

    total_tasks = 0
    total_samples = 0

    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        collapse_padding=True,
    )
    table.add_column("Task", style="bold")
    table.add_column("Samples", justify="right", style="yellow")
    table.add_column("Description", style="dim")

    for category, tasks in categories.items():
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        style = CATEGORY_STYLES.get(category, "white")

        # Category header row
        table.add_row()
        table.add_row(
            Text(f"  {display_name}", style=f"bold {style} underline"),
            "",
            "",
        )

        for task_name in tasks:
            count = all_configs[task_name].get("num_samples", "?")
            short_name = task_name.split("/")[1]
            description = TASK_DESCRIPTIONS.get(task_name, "")
            table.add_row(f"    {short_name}", str(count), description)
            total_tasks += 1
            if isinstance(count, int):
                total_samples += count

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"  [bold]{total_tasks}[/bold] tasks, "
        f"[bold]{total_samples}[/bold] samples  "
        f"[dim]Hub: {DEFAULT_HUB_REPO}[/dim]"
    )
    console.print()


def _print_sample(sample, index):
    """Print a single sample with Rich Panel formatting."""
    messages = sample["input"]["messages"]
    prompt = messages[-1]["content"]

    # Build content sections
    sections = []

    # Prompt
    sections.append(Text("Prompt", style="bold cyan"))
    sections.append(Text(prompt))

    # ICL examples (if any)
    if len(messages) > 1:
        num_icl = (len(messages) - 1) // 2
        sections.append(Text(f"\nIn-Context Examples ({num_icl})", style="bold cyan"))

    # Label
    sections.append(Text("\nLabel", style="bold cyan"))
    if isinstance(sample["label"], (dict, list)):
        label_str = json.dumps(sample["label"], indent=2, ensure_ascii=False)
    else:
        label_str = str(sample["label"])
    sections.append(Text(label_str))

    # Metadata
    sections.append(Text("\nMetadata", style="bold cyan"))
    if isinstance(sample["metadata"], (dict, list)):
        metadata_str = json.dumps(sample["metadata"], indent=2, ensure_ascii=False)
    else:
        metadata_str = str(sample["metadata"])
    sections.append(Text(metadata_str, style="dim"))

    # Combine into a single renderable
    content = Text("\n").join(sections)

    panel = Panel(
        content,
        title=f"[bold]Sample {index}[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def _stream_samples_from_hub(task_name, num_samples):
    """Stream samples from HuggingFace Hub without downloading the full dataset."""
    from datasets import load_dataset

    from parallelbench.datasets.task import _try_json_loads, task_name_to_config_name

    config_name = task_name_to_config_name(task_name)
    stream = load_dataset(DEFAULT_HUB_REPO, config_name, split="test", streaming=True)

    samples = []
    for row in itertools.islice(stream, num_samples):
        prompt_template = row["prompt"]
        input_data = _try_json_loads(row["input"])
        answer = _try_json_loads(row["answer"])
        metadata = _try_json_loads(row["metadata"])

        prompt = prompt_template.format(**input_data).replace("\\n", "\n")
        messages = [{"role": "user", "content": prompt}]

        samples.append(
            {
                "input": {"messages": messages},
                "label": answer,
                "metadata": metadata,
            }
        )

    return samples


def _get_available_task_names():
    """Get available task names from local configs (no heavy imports)."""
    all_configs = _load_all_task_configs()
    return sorted(name for name in all_configs if not name.startswith("_"))


def _print_task_samples(task_name, index=None, num_samples=3, generate=False):
    """Print samples from a specific task."""
    available = _get_available_task_names()
    if task_name not in available:
        console.print(
            f"\n[bold red]Error:[/bold red] Unknown task [yellow]'{task_name}'[/yellow]"
        )
        console.print("\nAvailable tasks:")
        for name in available:
            console.print(f"  [green]{name}[/green]")
        console.print()
        return

    description = TASK_DESCRIPTIONS.get(task_name, "")
    console.print()
    console.print(f"  [bold]Task:[/bold]   {task_name}")
    if description:
        console.print(f"  [bold]About:[/bold]  {description}")

    required = (index + 1) if index is not None else num_samples

    if generate:
        with console.status("[bold green]Generating samples locally..."):
            from parallelbench.datasets import ParallelBench

            dataset = ParallelBench(task_name, flex_config={"num_samples": required})
            samples = [dataset[i] for i in range(len(dataset))]
        console.print("  [bold]Source:[/bold] generated")
    else:
        with console.status(f"[bold green]Loading from {DEFAULT_HUB_REPO}..."):
            samples = _stream_samples_from_hub(task_name, required)
        console.print(f"  [bold]Source:[/bold] {DEFAULT_HUB_REPO}")

    console.print()

    if index is not None:
        if index >= len(samples):
            console.print(
                f"[bold red]Error:[/bold red] Index {index} out of range "
                f"(only {len(samples)} samples loaded)"
            )
            return
        _print_sample(samples[index], index)
    else:
        for i, sample in enumerate(samples):
            _print_sample(sample, i)


def main():
    parser = argparse.ArgumentParser(
        description="Browse ParallelBench tasks and samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              pb browse                              List all tasks
              pb browse waiting_line/copy            Show first 3 samples from Hub
              pb browse waiting_line/copy -n 5       Show first 5 samples
              pb browse waiting_line/copy --index 3  Show sample at index 3
              pb browse waiting_line/copy --generate Use locally generated data
        """),
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task name to browse (e.g., waiting_line/copy)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Show a specific sample by index",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to show (default: 3)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate samples locally instead of loading from HuggingFace Hub",
    )

    args = parser.parse_args()

    if args.task is None:
        _print_task_list()
    else:
        _print_task_samples(
            args.task,
            index=args.index,
            num_samples=args.num_samples,
            generate=args.generate,
        )
