#!/usr/bin/env bash
# Run Dream 7B with KLASS.
#
# TODO: Define sweep parameters for klass.
#
# Usage:
#   bash scripts/dream/dream-7b/klass.sh                    # single GPU
#   bash scripts/dream/dream-7b/klass.sh --num_processes 2  # multi GPU

set -euo pipefail

EXTRA_ARGS="${*}"
OUTPUT_DIR="results"
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

# TODO: Define sweep parameter and values for klass
echo "ERROR: Sweep parameters not yet defined for klass. Please update this script."
exit 1
