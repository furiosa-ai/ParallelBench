#!/usr/bin/env bash
# Run LLaDA 1.5 with EB-Sampler.
#
# TODO: Define sweep parameters for eb_sampler.
#
# Usage:
#   bash scripts/llada/llada-1.5/eb_sampler.sh                    # single GPU
#   bash scripts/llada/llada-1.5/eb_sampler.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for eb_sampler
echo "ERROR: Sweep parameters not yet defined for eb_sampler. Please update this script."
exit 1
