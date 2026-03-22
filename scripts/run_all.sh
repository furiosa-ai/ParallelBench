#!/usr/bin/env bash
# Run all implemented Dream and LLaDA family experiments.
#
# Skips scripts with unimplemented methods (TODO sweep parameters):
#   - dream/dream-7b/apd.sh
#   - dream/dream-7b/eb_sampler.sh
#   - llada/llada-1.5/eb_sampler.sh
#   - llada/llada-1.5/pc_sampler_confidence.sh
#   - llada/llada-1.5/pc_sampler_random.sh
#
# Usage:
#   bash scripts/run_all.sh                    # single GPU
#   bash scripts/run_all.sh --num_processes 2  # multi GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS="${*}"

# Share a single run name across all experiments
export PB_RUN_NAME="${PB_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_all}"

SCRIPTS=(
    # ── Dream 7B ──
    dream/dream-7b/confidence_factor.sh
    dream/dream-7b/confidence_threshold.sh
    dream/dream-7b/confidence_topk.sh
    dream/dream-7b/entropy_topk.sh
    dream/dream-7b/klass.sh
    dream/dream-7b/left_to_right.sh
    dream/dream-7b/random.sh
    dream/dream-7b/topk_margin.sh

    # ── DiffuCoder ──
    dream/diffucoder/confidence_factor.sh
    dream/diffucoder/confidence_threshold.sh
    dream/diffucoder/confidence_topk.sh
    dream/diffucoder/entropy_topk.sh
    dream/diffucoder/left_to_right.sh
    dream/diffucoder/random.sh
    dream/diffucoder/topk_margin.sh

    # ── LLaDA 1.5 ──
    llada/llada-1.5/confidence_factor.sh
    llada/llada-1.5/confidence_threshold.sh
    llada/llada-1.5/confidence_topk.sh
    llada/llada-1.5/dus.sh
    llada/llada-1.5/entropy_topk.sh
    llada/llada-1.5/klass.sh
    llada/llada-1.5/left_to_right.sh
    llada/llada-1.5/random.sh
    llada/llada-1.5/slowfast.sh
    llada/llada-1.5/topk_margin.sh
    llada/llada-1.5/wino_dllm.sh

    # ── LLaDA 8B Instruct ──
    llada/llada-8b-instruct/confidence_factor.sh
    llada/llada-8b-instruct/confidence_threshold.sh
    llada/llada-8b-instruct/confidence_topk.sh
    llada/llada-8b-instruct/entropy_topk.sh
    llada/llada-8b-instruct/left_to_right.sh
    llada/llada-8b-instruct/random.sh
    llada/llada-8b-instruct/topk_margin.sh
)

TOTAL=${#SCRIPTS[@]}
CURRENT=0
FAILED=()

for script in "${SCRIPTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    FULL_PATH="${SCRIPT_DIR}/${script}"

    echo ""
    echo "############################################################"
    echo "# [${CURRENT}/${TOTAL}] ${script}"
    echo "############################################################"

    if bash "${FULL_PATH}" ${EXTRA_ARGS}; then
        echo "[${CURRENT}/${TOTAL}] ${script} ... done"
    else
        echo "[${CURRENT}/${TOTAL}] ${script} ... FAILED (exit code $?)"
        FAILED+=("${script}")
    fi
done

echo ""
echo "============================================================"
echo "All experiments finished. (${TOTAL} total)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    exit 1
else
    echo "All experiments succeeded."
fi
