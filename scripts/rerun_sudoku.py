#!/usr/bin/env python3
"""Rerun sudoku experiments and merge results into existing result directories.

Usage:
    # Step 1: Regenerate data and push to Hub (one-time)
    pb data --split test --push --repo_id furiosa-ai/ParallelBench

    # Step 2: Dry run — show what commands would be executed
    python scripts/rerun_sudoku.py --results_dir results/ --dry_run

    # Step 3: Run for a specific model first (to verify)
    python scripts/rerun_sudoku.py --results_dir results/ --filter_model GSAI-ML/LLaDA-1.5

    # Step 4: Run all
    python scripts/rerun_sudoku.py --results_dir results/

    # Step 5: Merge only (if eval already completed)
    python scripts/rerun_sudoku.py --results_dir results/ --merge_only --temp_dir /tmp/sudoku_rerun
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SUDOKU_TASK = "parallelbench_puzzles_sudoku_n4"
PUZZLES_GROUP = "parallelbench_puzzles"
ALL_GROUP = "parallelbench_all"
# All individual tasks (17 total) — used for recomputing averages
CATEGORY_GROUPS = [
    PUZZLES_GROUP,
    ALL_GROUP,
    "parallelbench_waiting_line",
    "parallelbench_text_writing",
]
AGGREGATE_METRICS = ["score,none", "nfe,none", "tokens_per_step,none"]


def discover_result_directories(results_dir: str) -> list[Path]:
    """Find all result directories containing results_parallelbench.json."""
    results_path = Path(results_dir)
    return sorted(results_path.rglob("results_parallelbench.json"))


def load_config(results_file: Path) -> dict:
    """Load config from a results JSON file."""
    with open(results_file) as f:
        data = json.load(f)
    return data["config"]


def build_gen_kwargs_string(gen_kwargs: dict) -> str:
    """Convert gen_kwargs dict to CLI string format."""
    parts = []
    for key, value in gen_kwargs.items():
        parts.append(f"{key}={value}")
    return ",".join(parts)


def build_eval_command(
    config: dict, output_path: str, num_processes: int | None = None
) -> list[str]:
    """Build the eval command for sudoku only, matching existing script patterns.

    Uses `accelerate launch` for multi-GPU support and includes
    --apply_chat_template, --fewshot_as_multiturn, --log_samples
    to match the original experiment scripts.
    """
    model = config["model"]
    model_args = config["model_args"]
    gen_kwargs = build_gen_kwargs_string(config["gen_kwargs"])
    batch_size = config.get("batch_size", "1")

    if num_processes and num_processes > 1:
        cmd = [
            "uv",
            "run",
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
            "-m",
            "parallelbench.cli.eval",
        ]
    else:
        cmd = ["uv", "run", "pb", "eval"]

    cmd.extend(
        [
            "--model",
            model,
            "--model_args",
            model_args,
            "--gen_kwargs",
            gen_kwargs,
            "--tasks",
            SUDOKU_TASK,
            "--include_path",
            "parallelbench/tasks",
            "--batch_size",
            str(batch_size),
            "--apply_chat_template",
            "--fewshot_as_multiturn",
            "--log_samples",
            "--output_path",
            output_path,
        ]
    )
    return cmd


def find_individual_tasks(results_data: dict) -> list[str]:
    """Find all individual task keys (not group keys) in results."""
    group_keys = set(CATEGORY_GROUPS)
    return [
        key
        for key in results_data["results"]
        if key not in group_keys and key.startswith("parallelbench_")
    ]


def get_tasks_in_group(group_name: str, individual_tasks: list[str]) -> list[str]:
    """Get individual tasks belonging to a group."""
    if group_name == ALL_GROUP:
        return individual_tasks
    # Group tasks share a prefix with the group name
    # e.g., parallelbench_puzzles_sudoku_n4 belongs to parallelbench_puzzles
    return [t for t in individual_tasks if t.startswith(group_name + "_")]


def recompute_group_average(
    results_data: dict, group_name: str, individual_tasks: list[str]
) -> None:
    """Recompute weighted average for a group (all tasks have equal weight=100)."""
    tasks_in_group = get_tasks_in_group(group_name, individual_tasks)
    if not tasks_in_group:
        return

    group_data = results_data["results"].get(group_name, {})
    for metric in AGGREGATE_METRICS:
        values = []
        for task_key in tasks_in_group:
            task_data = results_data["results"].get(task_key, {})
            val = task_data.get(metric)
            if val is not None and not (
                isinstance(val, float) and val != val
            ):  # skip NaN
                values.append(val)
        if values:
            group_data[metric] = sum(values) / len(values)

    results_data["results"][group_name] = group_data


def merge_sudoku_results(original_dir: Path, new_results_dir: Path) -> bool:
    """Merge new sudoku results into the original result directory.

    Returns True if merge was successful.
    """
    original_results_file = original_dir / "results_parallelbench.json"
    original_samples_file = original_dir / f"samples_{SUDOKU_TASK}.jsonl"

    # Find new results
    new_results_files = list(new_results_dir.rglob("results_parallelbench.json"))
    if not new_results_files:
        print(f"  WARNING: No new results found in {new_results_dir}")
        return False

    new_results_file = new_results_files[0]
    new_samples_file = new_results_file.parent / f"samples_{SUDOKU_TASK}.jsonl"

    if not new_samples_file.exists():
        print(f"  WARNING: No new sudoku samples found: {new_samples_file}")
        return False

    # Load both results
    with open(original_results_file) as f:
        original_data = json.load(f)
    with open(new_results_file) as f:
        new_data = json.load(f)

    # 1. Replace sudoku task metrics
    if SUDOKU_TASK not in new_data["results"]:
        print(f"  WARNING: {SUDOKU_TASK} not in new results")
        return False

    original_data["results"][SUDOKU_TASK] = new_data["results"][SUDOKU_TASK]

    # 2. Recompute group averages
    individual_tasks = find_individual_tasks(original_data)
    recompute_group_average(original_data, PUZZLES_GROUP, individual_tasks)
    recompute_group_average(original_data, ALL_GROUP, individual_tasks)

    # 3. Write updated results JSON
    with open(original_results_file, "w") as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)

    # 4. Replace sudoku samples JSONL
    shutil.copy2(new_samples_file, original_samples_file)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rerun sudoku experiments and merge results"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/", help="Root results directory"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for new eval results (default: auto)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Only print commands, don't execute"
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Skip eval, only merge existing temp results",
    )
    parser.add_argument(
        "--filter_model",
        type=str,
        default=None,
        help="Only process runs for this model (e.g., GSAI-ML/LLaDA-1.5)",
    )
    parser.add_argument(
        "--skip_eval_errors",
        action="store_true",
        help="Continue on eval errors instead of stopping",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of GPUs for accelerate launch (default: single GPU)",
    )
    args = parser.parse_args()

    # Discover all result directories
    results_files = discover_result_directories(args.results_dir)
    print(f"Found {len(results_files)} result directories")

    # Set up temp directory
    if args.temp_dir:
        temp_base = Path(args.temp_dir)
        temp_base.mkdir(parents=True, exist_ok=True)
    else:
        temp_base = Path(tempfile.mkdtemp(prefix="sudoku_rerun_"))
    print(f"Temp directory: {temp_base}")

    # Group runs by (model, gen_kwargs) to deduplicate eval calls.
    # Multiple result dirs with the same config only need one eval run.
    # Key: (model, model_args, gen_kwargs_str) → list of (results_file, config, original_dir)
    config_groups: dict[str, list[tuple[Path, dict, Path]]] = {}
    skipped = 0
    for rf in results_files:
        config = load_config(rf)
        model_name = (
            config["model_args"].split("=", 1)[1]
            if "=" in config["model_args"]
            else config["model_args"]
        )

        if args.filter_model and model_name != args.filter_model:
            skipped += 1
            continue

        gen_kwargs_str = build_gen_kwargs_string(config["gen_kwargs"])
        group_key = f"{config['model']}|{config['model_args']}|{gen_kwargs_str}"
        config_groups.setdefault(group_key, []).append((rf, config, rf.parent))

    total_dirs = sum(len(dirs) for dirs in config_groups.values())
    unique_configs = len(config_groups)
    print(
        f"Processing {total_dirs} result dirs ({unique_configs} unique configs, {total_dirs - unique_configs} deduped)"
    )
    if skipped:
        print(f"Skipped {skipped} dirs (filtered)")

    if args.dry_run:
        print("\n=== DRY RUN — Commands that would be executed ===\n")

    # Process each unique config
    eval_idx = 0
    failed = []
    for group_key, entries in sorted(config_groups.items()):
        eval_idx += 1
        config = entries[0][1]  # All entries share the same config
        model_name = (
            config["model_args"].split("=", 1)[1]
            if "=" in config["model_args"]
            else config["model_args"]
        )
        gen_kwargs_str = build_gen_kwargs_string(config["gen_kwargs"])

        # Single temp output per unique config
        safe_key = (
            group_key.replace("/", "_")
            .replace("|", "__")
            .replace(" ", "_")
            .replace(",", "_")
        )
        temp_output = temp_base / safe_key

        print(f"\n[{eval_idx}/{unique_configs}] {model_name} | {gen_kwargs_str}")
        print(f"  Applies to {len(entries)} result dir(s)")

        if not args.merge_only:
            cmd = build_eval_command(config, str(temp_output), args.num_processes)

            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
                for _, _, original_dir in entries:
                    print(f"    → merge into: {original_dir}")
                continue

            # Run evaluation (once per unique config)
            print("  Running eval...")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).resolve().parent.parent),
                )
                if result.returncode != 0:
                    print(f"  ERROR (exit {result.returncode}):")
                    print(
                        f"  {result.stderr[-500:]}"
                        if result.stderr
                        else "  (no stderr)"
                    )
                    if not args.skip_eval_errors:
                        print("  Stopping. Use --skip_eval_errors to continue.")
                        sys.exit(1)
                    for _, _, d in entries:
                        failed.append(str(d))
                    continue
                print("  Eval complete.")
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                if not args.skip_eval_errors:
                    sys.exit(1)
                for _, _, d in entries:
                    failed.append(str(d))
                continue

        # Merge results into all matching directories
        if args.dry_run:
            continue

        for _, _, original_dir in entries:
            print(f"  Merging into {original_dir.name}...")
            if merge_sudoku_results(original_dir, temp_output):
                print("    OK")
            else:
                print("    FAILED")
                failed.append(str(original_dir))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total result dirs: {total_dirs}")
    print(f"Unique eval configs: {unique_configs}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed directories:")
        for d in failed:
            print(f"  - {d}")
    print(f"Temp directory: {temp_base}")
    if not args.dry_run and not failed:
        print("\nAll sudoku results updated successfully!")
        print(f"You can remove the temp directory: rm -rf {temp_base}")


if __name__ == "__main__":
    main()
