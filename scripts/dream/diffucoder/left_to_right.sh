#!/usr/bin/env bash
# Run DiffuCoder with left-to-right.
#
# TODO: Define sweep parameters for left_to_right.
#
# Usage:
#   bash scripts/dream/diffucoder/left_to_right.sh                    # single GPU
#   bash scripts/dream/diffucoder/left_to_right.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for left_to_right
echo "ERROR: Sweep parameters not yet defined for left_to_right. Please update this script."
exit 1
