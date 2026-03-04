"""Wrapper around lm_eval CLI that pre-registers ParallelBench model wrappers."""

import parallelbench.lm_eval_models  # noqa: F401 — triggers @register_model

from lm_eval.__main__ import cli_evaluate


def main():
    cli_evaluate()


if __name__ == "__main__":
    main()
