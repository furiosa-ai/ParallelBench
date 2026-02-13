"""
Dry-run validation script for cfg YAML files.

Validates that each config entry's `generation` dict is compatible with
the GenerationConfig dataclass that the corresponding model would use at runtime.
This catches field name mismatches, unexpected kwargs, and validation errors
WITHOUT loading any models or requiring GPU.

Usage:
    # Validate a single list yaml:
    python validate_configs.py cfg/paper/benchmark/llada_1_5_all_tasks_list.yaml

    # Validate all cfg/paper files:
    python validate_configs.py cfg/paper/benchmark/*.yaml cfg/paper/dllm_vs_llm/*.yaml

    # Validate and show suggested fixes:
    python validate_configs.py --suggest-fix cfg/paper/benchmark/*.yaml
"""

import argparse
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Import all model modules to populate ModelRegistry
from model import (  # noqa: F401
    AnthropicModel,
    DreamModel,
    LladaModel,
    MercuryModel,
    SeddModel,
    TradoModel,
    TransformersModel,
    vllmModel,
)
from model.api.anthropic_model import AnthropicGenerationConfig
from model.api.mercury_model import MercuryGenerationConfig
from model.local.dream.dream_model import DreamGenerationConfig
from model.local.llada.llada_model import LladaGenerationConfig
from model.local.sedd.sedd_model import SeddGenerationConfig
from model.local.trado.trado_model import TradoGenerationConfig
from model.local.transformers_model import TransformersGenerationConfig
from model.local.vllm_model import vllmGenerationConfig
from model.registry import ModelRegistry


# Known field name migrations (old_name -> new_name)
# Used for --suggest-fix mode
FIELD_MIGRATIONS = {
    "alg_threshold": "alg_threshold",
    "alg_factor": "alg_factor",
    "fast_dllm_use_cache": "use_fast_dllm_cache",
    "fast_dllm_dual_cache": "use_fast_dllm_dual_cache",
}


def get_generation_config_class(model_name, accel_framework):
    """
    Resolve model_name to the appropriate GenerationConfig class,
    replicating the dispatch logic in each model's generate() method.

    Returns:
        tuple: (GenerationConfig class, bool: whether accel_framework should be passed)
    """
    # Try ModelRegistry first (same as model/__init__.py::load_model)
    try:
        model_class = ModelRegistry.get_model_class(model_name)
    except ValueError:
        model_class = None

    if model_class is not None:
        # Map model class -> generation config class
        class_name = model_class.__name__
        config_map = {
            "LladaModel": (LladaGenerationConfig, True),
            "DreamModel": (DreamGenerationConfig, True),
            "TradoModel": (TradoGenerationConfig, True),
            "SeddModel": (SeddGenerationConfig, True),
            "MercuryModel": (MercuryGenerationConfig, False),
            "AnthropicModel": (AnthropicGenerationConfig, False),
        }
        if class_name in config_map:
            return config_map[class_name]
        raise ValueError(
            f"Model class '{class_name}' matched but has no known GenerationConfig mapping."
        )

    # Fallback: accel_framework-based dispatch
    if accel_framework == "vllm":
        return vllmGenerationConfig, False
    elif accel_framework == "transformers":
        return TransformersGenerationConfig, False

    raise ValueError(
        f"Cannot resolve model_name='{model_name}' with accel_framework='{accel_framework}' "
        f"to a GenerationConfig class."
    )


def get_accepted_fields(config_class):
    """Get the set of field names accepted by a dataclass."""
    return {field.name for field in dataclasses.fields(config_class)}


def validate_single_config(config_entry, entry_index, file_path, suggest_fix=False):
    """
    Validate a single config entry from a list YAML.

    Returns:
        list of error dicts, each with:
            - entry_index: int
            - model_name: str
            - error_type: 'unknown_field' | 'validation_error' | 'resolution_error'
            - message: str
            - suggestion: str (optional, if suggest_fix=True)
    """
    errors = []

    model_cfg = config_entry.get("model", {})
    model_name = model_cfg.get("model_name", "<unknown>")
    accel_framework = model_cfg.get("accel_framework", None)
    generation = config_entry.get("generation", {})
    dataset_cfg = config_entry.get("dataset", {})
    task_name = dataset_cfg.get("task", "<unknown>")

    if not generation:
        return errors  # No generation config to validate

    # Step 1: Resolve model_name -> GenerationConfig class
    try:
        config_class, needs_accel_framework = get_generation_config_class(
            model_name, accel_framework
        )
    except ValueError as e:
        errors.append(
            {
                "entry_index": entry_index,
                "model_name": model_name,
                "task": task_name,
                "error_type": "resolution_error",
                "message": str(e),
            }
        )
        return errors

    # Step 2: Check for unknown fields
    accepted_fields = get_accepted_fields(config_class)
    generation_keys = set(generation.keys())
    unknown_fields = generation_keys - accepted_fields

    if needs_accel_framework:
        # accel_framework is passed separately (not in the generation dict)
        # so it's fine if it's not in accepted_fields
        unknown_fields.discard("accel_framework")

    for field_name in sorted(unknown_fields):
        error = {
            "entry_index": entry_index,
            "model_name": model_name,
            "task": task_name,
            "error_type": "unknown_field",
            "message": (
                f"Unknown field '{field_name}' for {config_class.__name__}. "
                f"Accepted fields: {sorted(accepted_fields)}"
            ),
        }
        if suggest_fix and field_name in FIELD_MIGRATIONS:
            new_name = FIELD_MIGRATIONS[field_name]
            if new_name in accepted_fields:
                error["suggestion"] = f"Rename '{field_name}' -> '{new_name}'"
            else:
                error["suggestion"] = (
                    f"'{field_name}' is deprecated. Migration target '{new_name}' "
                    f"is not accepted by {config_class.__name__} either — remove this field."
                )
        errors.append(error)

    # Step 3: Try to instantiate the config (catches __post_init__ validation errors)
    # Only attempt if there are no unknown fields (otherwise TypeError will mask the real issue)
    if not unknown_fields:
        try:
            kwargs = dict(generation)
            if needs_accel_framework:
                kwargs["accel_framework"] = accel_framework
            config_class(**kwargs)
        except (TypeError, ValueError, AssertionError) as e:
            errors.append(
                {
                    "entry_index": entry_index,
                    "model_name": model_name,
                    "task": task_name,
                    "error_type": "validation_error",
                    "message": f"{type(e).__name__}: {e}",
                }
            )

    return errors


def validate_config_file(file_path, suggest_fix=False):
    """
    Validate all entries in a *_list.yaml config file.

    Returns:
        list of error dicts
    """
    path = Path(file_path)
    if not path.exists():
        return [
            {
                "entry_index": -1,
                "model_name": "",
                "task": "",
                "error_type": "file_error",
                "message": f"File not found: {file_path}",
            }
        ]

    with open(path) as f:
        configs = yaml.safe_load(f)

    if configs is None:
        return [
            {
                "entry_index": -1,
                "model_name": "",
                "task": "",
                "error_type": "file_error",
                "message": f"Empty or invalid YAML: {file_path}",
            }
        ]

    if not isinstance(configs, list):
        # Single config (non-list yaml)
        configs = [configs]

    all_errors = []
    for i, entry in enumerate(configs):
        entry_errors = validate_single_config(
            entry, i, file_path, suggest_fix=suggest_fix
        )
        all_errors.extend(entry_errors)

    return all_errors


def format_report(file_path, errors):
    """Format errors for a single file into a human-readable report."""
    lines = []
    if not errors:
        lines.append(f"  ✅ {file_path}: All {0} entries valid")
        return "\n".join(lines)

    # Group errors by type for summary
    by_type = defaultdict(list)
    for err in errors:
        by_type[err["error_type"]].append(err)

    lines.append(f"  ❌ {file_path}: {len(errors)} error(s)")

    # Deduplicate error messages for compact output
    seen_messages = set()
    for err in errors:
        key = (err["error_type"], err["message"])
        if key in seen_messages:
            continue
        seen_messages.add(key)

        # Count how many entries have this exact error
        count = sum(1 for e in errors if (e["error_type"], e["message"]) == key)
        prefix = f"     [{err['error_type']}]"

        if count > 1:
            lines.append(f"{prefix} (x{count}) {err['message']}")
        else:
            lines.append(f"{prefix} entry #{err['entry_index']}: {err['message']}")

        if "suggestion" in err:
            lines.append(f"       💡 Fix: {err['suggestion']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run validation for ParallelBench cfg YAML files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to *_list.yaml config files to validate.",
    )
    parser.add_argument(
        "--suggest-fix",
        action="store_true",
        help="Show suggested fixes for known field name migrations.",
    )
    args = parser.parse_args()

    total_errors = 0
    total_files = 0
    total_entries = 0

    print("=" * 70)
    print("ParallelBench Config Validation Report")
    print("=" * 70)

    for file_path in sorted(args.files):
        total_files += 1

        # Count entries
        with open(file_path) as f:
            configs = yaml.safe_load(f)
        if isinstance(configs, list):
            num_entries = len(configs)
        else:
            num_entries = 1
        total_entries += num_entries

        errors = validate_config_file(file_path, suggest_fix=args.suggest_fix)
        total_errors += len(errors)

        if errors:
            report = format_report(file_path, errors)
        else:
            report = f"  ✅ {file_path}: All {num_entries} entries valid"

        print(report)

    print("=" * 70)
    print(
        f"Summary: {total_files} file(s), {total_entries} entries, "
        f"{total_errors} error(s)"
    )

    if total_errors > 0:
        print(
            f"\n⚠️  {total_errors} validation error(s) found. Run with --suggest-fix for migration hints."
        )
        sys.exit(1)
    else:
        print("\n✅ All configs are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
