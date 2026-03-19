"""Wrapper around lm_eval CLI that pre-registers ParallelBench model wrappers
and uses a custom EvaluationTracker for gen_kwargs-based output organization."""

import sys
import uuid
from datetime import datetime


def _extract_run_name() -> str:
    """Extract --run_name from sys.argv, removing it before passing to lm-eval.

    lm-eval rejects unknown arguments, so we must pop --run_name before
    cli_evaluate() parses sys.argv.

    Returns:
        The run_name string: either the user-provided value or a timestamp+suffix
        in the format "YYYYMMDD_HHMMSS_XXXX" (e.g., "20260319_143052_a3f2").
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
