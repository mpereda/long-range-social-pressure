#!/bin/bash
# run_full_revision_pipeline.sh — PRE major-revision resweep (2026-09).
# Runs the corrected production sweep (notebooks 01-05, in validation-first
# order) and then, using the corrected results, the network-realization
# robustness study (Referee 1 #3 + Referee 2 #7).
#
# Assumes data/*.csv from the OLD (buggy stopping-criterion) run have
# already been archived and removed, so notebooks regenerate from scratch.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# Keeps the Mac awake for the whole pipeline (run_overnight.sh's own
# caffeinate only covers notebooks 01-05, not the robustness study after it).
caffeinate -i &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

echo "=============================================="
echo "Full revision pipeline start: $(date)"
echo "=============================================="

bash run_overnight.sh

echo ""
echo ">>> Production sweep done. Starting network-realization robustness study ($(date))"
python3 robustness_checks/network_realization_robustness.py 2>&1 | tee -a robustness_checks/network_realization_robustness.log

echo ""
echo "=============================================="
echo "Full revision pipeline done: $(date)"
echo "=============================================="
