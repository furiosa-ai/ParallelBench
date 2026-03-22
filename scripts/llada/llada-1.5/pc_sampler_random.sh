#!/usr/bin/env bash
# Run LLaDA 1.5 with PC-Sampler (Random).
#
# TODO: Define sweep parameters for pc_sampler_random.
#
# Usage:
#   bash scripts/llada/llada-1.5/pc_sampler_random.sh                    # single GPU
#   bash scripts/llada/llada-1.5/pc_sampler_random.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for pc_sampler_random
echo "ERROR: Sweep parameters not yet defined for pc_sampler_random. Please update this script."
exit 1
