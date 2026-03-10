"""Wrapper around lm_eval CLI that pre-registers ParallelBench model wrappers
and uses a custom EvaluationTracker for gen_kwargs-based output organization."""


def main():
    import parallelbench.lm_eval_wrappers  # noqa: F401 — triggers @register_model

    # Monkey-patch lm-eval's EvaluationTracker with our custom subclass
    # that organizes results into gen_kwargs subdirectories.
    # This must happen before cli_evaluate() imports and instantiates the tracker.
    import lm_eval.__main__ as _lm_eval_main

    from parallelbench.lm_eval_wrappers.evaluation_tracker import (
        ParallelBenchEvaluationTracker,
    )

    _lm_eval_main.EvaluationTracker = ParallelBenchEvaluationTracker
    _lm_eval_main.cli_evaluate()


if __name__ == "__main__":
    main()
