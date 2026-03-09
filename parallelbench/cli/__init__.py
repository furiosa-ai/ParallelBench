"""ParallelBench CLI.

Usage:
    pb eval [lm-eval options]     Run lm-eval evaluation
    pb data [options]             Generate benchmark data
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: pb <command> [options]")
        print()
        print("Commands:")
        print("  eval    Run lm-eval evaluation")
        print("  data    Generate benchmark data")
        print()
        print("Run 'pb <command> --help' for more information.")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "eval":
        from parallelbench.cli.eval import main as eval_main

        eval_main()
    elif command == "data":
        from parallelbench.datasets.generate import main as data_main

        data_main()
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: eval, data")
        sys.exit(1)


if __name__ == "__main__":
    main()
