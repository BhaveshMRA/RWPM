#!/bin/bash
# Runs for the Phase 3 feedback (macOS or Linux), all in one go (~2 h):
#   caffeinate -i bash code/run_feedback_mac.sh
# Needs gitskills/work_full from the full run. Every step is logged to gitskills/feedback_log.txt and
# the script stops at the first error.
#  1. final model (k = 5): step 6 saves the confusion matrix of every ablation (Fix 4, same-owner
#     shortcut), step 16 counts the owners behind the largest components (small point)
#  2. recall kit stratified by band (Fix 1), written to labeling/recall/
#  3. boilerplate threshold sensitivity on the gold set (Fix 3): by name k = 50, by owner k = 5 and
#     k = 410, each in its own folders; white-box importances for every threshold
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE_WORK:-$REPO/gitskills/work_full}
LOG=$REPO/gitskills/feedback_log.txt
cd "$REPO/code"
source ../.venv/bin/activate
python -m pip install -q -r ../requirements-full.txt
run() { echo "=== $(date '+%a %H:%M:%S') : $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
[ -f "$BASE/features_all.pkl" ] || { echo "=== $BASE/features_all.pkl not found" | tee -a "$LOG"; exit 1; }
echo "=== feedback run started $(date)" | tee -a "$LOG"

export SKILL_WORK=$BASE SKILL_RESULTS=$REPO/paper/results_full
run python 06_rq1_rq2.py
run python 16_edge_diagnostics.py
run python 19_recall_kit.py --make

variant() {
  local name=$1; shift
  local W=$REPO/gitskills/work_full_$name R=$REPO/paper/results_full_$name
  mkdir -p "$W" "$R"
  for f in full_skills.db skills.pkl occurrences.pkl population_occurrences.pkl candidates_full.npz minhash.npy; do
    ln -sf "$BASE/$f" "$W/$f"
  done
  export SKILL_WORK=$W SKILL_RESULTS=$R
  run python 11_full_index.py --boiler-only "$@"
  run python 13_full_features.py --pairs gold
  run python 06_rq1_rq2.py
  run python 09_whitebox.py
}
variant k50 --boiler-k 50
variant own5 --boiler-by owner --boiler-k 5
variant own410 --boiler-by owner --boiler-k 410
export SKILL_WORK=$REPO/gitskills/work_full_k410 SKILL_RESULTS=$REPO/paper/results_full_k410
run python 09_whitebox.py
echo "=== feedback run finished $(date)" | tee -a "$LOG"
