"""Wrapper around lm_eval CLI that pre-registers ParallelBench model wrappers
and uses a custom EvaluationTracker for gen_kwargs-based output organization."""

import os
import sys
import uuid
from datetime import datetime


def _extract_run_name() -> str:
    """Extract --run_name from sys.argv, removing it before passing to lm-eval.

    lm-eval rejects unknown arguments, so we must pop --run_name before
    cli_evaluate() parses sys.argv.

    Resolution order:
        1. ``--run_name`` CLI flag (highest priority)
        2. ``PB_RUN_NAME`` environment variable
        3. Auto-generated ``YYYYMMDD_HHMMSS_XXXX`` timestamp

    The ``PB_RUN_NAME`` env var is useful for grouping results from multiple
    ``pb eval`` invocations into the same run directory (e.g., when running
    category-specific scripts separately).
    """
    run_name = None
    args = sys.argv[:]
    i = 0
    while i < len(args):
        if args[i] == "--run_name":
            if i + 1 < len(args):
                run_name = args[i + 1]
                del args[i : i + 2]
            else:
                del args[i]
        elif args[i].startswith("--run_name="):
            run_name = args[i].split("=", 1)[1]
            del args[i]
        else:
            i += 1
    sys.argv = args

    if run_name is None:
        run_name = os.environ.get("PB_RUN_NAME")
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    return run_name


def main():
    import parallelbench.lm_eval_wrappers  # noqa: F401 — triggers @register_model

    # Monkey-patch lm-eval's EvaluationTracker with our custom subclass
    # that organizes results into gen_kwargs subdirectories.
    # This must happen before cli_evaluate() imports and instantiates the tracker.
    import lm_eval.__main__ as _lm_eval_main

    from parallelbench.lm_eval_wrappers.evaluation_tracker import (
        ParallelBenchEvaluationTracker,
    )

    run_name = _extract_run_name()
    ParallelBenchEvaluationTracker.run_name = run_name

    _lm_eval_main.EvaluationTracker = ParallelBenchEvaluationTracker
    _lm_eval_main.cli_evaluate()


if __name__ == "__main__":
    main()
