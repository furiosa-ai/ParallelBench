"""Wrapper around lm_eval CLI that pre-registers ParallelBench model wrappers
and uses a custom EvaluationTracker for gen_kwargs-based output organization."""

import sys
import uuid


def _extract_run_id() -> str:
    """Extract --run_id from sys.argv, removing it before passing to lm-eval.

    lm-eval rejects unknown arguments, so we must pop --run_id before
    cli_evaluate() parses sys.argv.

    Returns:
        The run_id string: either the user-provided value or an 8-char hex UUID.
    """
    run_id = None
    args = sys.argv[:]
    i = 0
    while i < len(args):
        if args[i] == "--run_id":
            if i + 1 < len(args):
                run_id = args[i + 1]
                del args[i : i + 2]
            else:
                del args[i]
        elif args[i].startswith("--run_id="):
            run_id = args[i].split("=", 1)[1]
            del args[i]
        else:
            i += 1
    sys.argv = args

    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    return run_id


def main():
    import parallelbench.lm_eval_wrappers  # noqa: F401 — triggers @register_model

    # Monkey-patch lm-eval's EvaluationTracker with our custom subclass
    # that organizes results into gen_kwargs subdirectories.
    # This must happen before cli_evaluate() imports and instantiates the tracker.
    import lm_eval.__main__ as _lm_eval_main

    from parallelbench.lm_eval_wrappers.evaluation_tracker import (
        ParallelBenchEvaluationTracker,
    )

    run_id = _extract_run_id()
    ParallelBenchEvaluationTracker.run_id = run_id

    _lm_eval_main.EvaluationTracker = ParallelBenchEvaluationTracker
    _lm_eval_main.cli_evaluate()


if __name__ == "__main__":
    main()
