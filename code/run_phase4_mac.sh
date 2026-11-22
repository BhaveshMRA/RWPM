#!/bin/bash
# Phase 4 full-data runs (macOS or Linux), all in one go:
#   caffeinate -i bash code/run_phase4_mac.sh [/path/to/agent_skills_release.db]
# Case A: gitskills/work_full from the Phase 3 run still exists -> only the new analyses run (~4 h).
#         The database is not needed in this case.
# Case B: it is gone -> steps 10-13 and 6 rebuild it first (~4 h more, needs the database); results go to
#         paper/results_full_rerun, so the committed paper/results_full is never overwritten.
# Every step is logged to gitskills/phase4_log.txt; the script stops at the first error.
set -euo pipefail
DB=${1:-}
if [ -n "$DB" ]; then DB=$(cd "$(dirname "$DB")" && pwd)/$(basename "$DB"); fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE_WORK:-$REPO/gitskills/work_full}
VAR=$REPO/gitskills/work_full_k410
LOG=$REPO/gitskills/phase4_log.txt
mkdir -p "$REPO/gitskills"
cd "$REPO/code"
source ../.venv/bin/activate
python -m pip install -q -r ../requirements-full.txt
run() { echo "=== $(date '+%a %H:%M:%S') : $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
echo "=== Phase 4 run started $(date); base folder $BASE" | tee -a "$LOG"

# case B: rebuild the base work folder if it is missing
if [ ! -f "$BASE/features_all.pkl" ]; then
  echo "=== base work folder not found: rebuilding it (case B)" | tee -a "$LOG"
  if [ -z "$DB" ] || [ ! -f "$DB" ]; then
    echo "=== case B needs the release database: bash code/run_phase4_mac.sh /path/to/agent_skills_release.db" | tee -a "$LOG"; exit 1
  fi
  mkdir -p "$BASE" "$REPO/paper/results_full_rerun"
  export SKILL_WORK=$BASE SKILL_RESULTS=$REPO/paper/results_full_rerun
  run python 10_full_extract.py --db "$DB"
  run python 11_full_index.py
  run python 12_full_candidates.py
  run python 13_full_features.py --pairs gold
  run python 13_full_features.py --pairs all --workers 4
  run python 06_rq1_rq2.py
  if cmp -s "$REPO/paper/results_full_rerun/rq1_oof_predictions.csv" "$REPO/paper/results_full/rq1_oof_predictions.csv"; then
    echo "=== rebuild check: out-of-fold predictions identical to the committed run" | tee -a "$LOG"
  else
    echo "=== rebuild check: out-of-fold predictions DIFFER from the committed run - send the log before using the results" | tee -a "$LOG"
  fi
fi

# new analyses on the final model (k = 5)
export SKILL_WORK=$BASE SKILL_RESULTS=$REPO/paper/results_full
run python 17_copy_network.py --all-links
run python 07_rq3.py --hub-degree 50
run python 18_stress_kit.py --make

# variant: boilerplate threshold scaled to the corpus (k = 410), in its own folders
mkdir -p "$VAR" "$REPO/paper/results_full_k410"
for f in full_skills.db skills.pkl occurrences.pkl population_occurrences.pkl candidates_full.npz minhash.npy; do
  ln -sf "$BASE/$f" "$VAR/$f"
done
export SKILL_WORK=$VAR SKILL_RESULTS=$REPO/paper/results_full_k410
run python 11_full_index.py --boiler-only --boiler-k 410
run python 13_full_features.py --pairs gold
run python 06_rq1_rq2.py
run python 08_stats.py
run python 13_full_features.py --pairs all --workers 4
run python 07_rq3.py
run python 07_rq3.py --same-name-only
run python 07_rq3.py --hub-degree 50
run python 16_edge_diagnostics.py
run python 17_copy_network.py --all-links --history "$BASE/seddik_history.parquet"
echo "=== Phase 4 run finished $(date)" | tee -a "$LOG"
