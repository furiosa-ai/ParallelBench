#!/usr/bin/env bash
# Run LLaDA 1.5 with WINO-DLLM.
#
# TODO: Define sweep parameters for wino_dllm.
#
# Usage:
#   bash scripts/llada/llada-1.5/wino_dllm.sh                    # single GPU
#   bash scripts/llada/llada-1.5/wino_dllm.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for wino_dllm
echo "ERROR: Sweep parameters not yet defined for wino_dllm. Please update this script."
exit 1
