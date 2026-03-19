#!/usr/bin/env bash
# Run LLaDA 1.5 with SlowFast Sampling.
#
# TODO: Define sweep parameters for slowfast.
#
# Usage:
#   bash scripts/llada/llada-1.5/slowfast.sh                    # single GPU
#   bash scripts/llada/llada-1.5/slowfast.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for slowfast
echo "ERROR: Sweep parameters not yet defined for slowfast. Please update this script."
exit 1
