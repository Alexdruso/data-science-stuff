#!/usr/bin/env bash
# Day-23 lane (GPU): 3 matched seeds of the `_dp` arm — realmlp_te13 + the label-free
# DRIVER-POSTERIOR block (dp_features.dp_block, 20 cols).
#
# WHY. Day-22's M-slope priced `imp` (sleep reconstruction, scalar point estimate) at
# exponent -0.850 = "not information". But `imp` reconstructs the WEAKEST driver: the
# train_sm.py diagnostic found physical_activity_level is by far the most recoverable
# (logloss gain +0.5515, top-1 0.712 vs 0.339) and it was never fed to a strong leg.
# This arm asks whether driver STRENGTH changes the exponent, or whether the whole
# reconstruction family is a variance reducer.
#
# PRE-REGISTERED BARS (comp CLAUDE.md Day-23 block; NOT to be moved):
#   exponent <= -0.060  => DEAD regardless of M=1 size. No escalation, no level read.
#   exponent >  0       => escalate to the deployed M and read the level.
#   A #2 swap additionally needs >= +0.001 at deployed M + slot + audit + byte verify.
#   A tie means NO swap.
# HONEST PRIOR: expect a clean negative (imp -0.850; train_dp.py already measured this
# channel at -0.0002 solo on an older recipe; the motivating diagnostic is in-sample and
# pooled, on a channel carrying the Day-4 train/test mask shift).
#
# Seeds 42/7/123 match the stock `realmlp_te13_u` arm and the `imp` arm exactly, so the
# exponent read is paired against both.
#
# Reboot-safe (per-fold checkpoints + skip-if-exists) per project_pc_random_reboots.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python

export S6E7_DP=1
# ⚠️ S6E7_TE13=1 IS LOAD-BEARING — omitting it is a SILENT RENAME, never an error
# (MODEL_NAME drops the "_te13" segment). The 2026-07-26 incident lost 50 min to exactly
# this; the post-run assert below is what makes recipe drift loud.
export S6E7_TE13=1
export PYTHONUNBUFFERED=1

LOCK=results/.dp_breadth.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_dp_breadth.sh holds $LOCK — refusing to start a second instance"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

for SEED in 42 7 123; do
  OUT="results/oof_realmlp_te13_dp_u_s${SEED}.npy"
  if [ -f "$OUT" ]; then
    echo "=== seed ${SEED}: already present, skipping ==="
    continue
  fi
  echo "=== seed ${SEED} starting $(date -Is) ==="
  S6E7_SEEDS="${SEED}" S6E7_RUN_TAG="_u_s${SEED}" \
    $PY src/train_realmlp_td.py 2>&1 \
    | tee "results/log_dp_s${SEED}.txt"
  if [ ! -f "$OUT" ]; then
    echo "FATAL: seed ${SEED} did not produce ${OUT}"
    echo "       (recipe drift — check S6E7_TE13/S6E7_DP). Aborting lane."
    exit 1
  fi
  echo "=== seed ${SEED} done $(date -Is) ==="
done
echo "=== dp breadth lane complete $(date -Is) ==="
