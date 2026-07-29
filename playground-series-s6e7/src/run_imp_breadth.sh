#!/usr/bin/env bash
# Day-22 lane (GPU): seeds 7/123/2024 of the PURE `imp` arm — realmlp_te13 + the
# driver-reconstruction block, at the STOCK config (no joint-TPE basin).
#
# WHY. `ji` = joint basin (a CONFIG change) x imp (an INFORMATION channel), and Day-21
# read its M-slope as -0.71, i.e. it dilutes. Day-22's validated M-slope instrument
# (probe_mslope.py) says a uniform MEAN SHIFT does NOT dilute (exponent -0.03) while a
# VARIANCE REDUCER does (-1.00). If `imp` is information and the basin is not, the two
# components should separate — and `ji`'s intermediate exponent would be the mixture.
# That is the experiment: `imp` alone, at 4 matched seeds, read by exponent.
#
# s42 already exists (results/oof_realmlp_te13_imp_u_s42.npy, Day-17). This adds 7/123/2024
# so the arm is matched with the 4-seed `ji` arm and the stock breadth.
#
# Reboot-safe (per-fold checkpoints + skip-if-exists) per project_pc_random_reboots.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python

export S6E7_IMP=1
# ⚠️ S6E7_TE13=1 IS LOAD-BEARING — omitting it is a SILENT RENAME, never an error
# (MODEL_NAME drops the "_te13" segment). The 2026-07-26 incident lost 50 min to exactly
# this; the post-run assert below is what makes recipe drift loud.
export S6E7_TE13=1
export PYTHONUNBUFFERED=1

LOCK=results/.imp_breadth.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_imp_breadth.sh holds $LOCK — refusing to start a second instance"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

for SEED in 7 123 2024; do
  if [ -f "results/oof_realmlp_te13_imp_u_s${SEED}.npy" ]; then
    echo "=== seed ${SEED}: already present, skipping ==="
    continue
  fi
  echo "=== seed ${SEED} starting $(date -Is) ==="
  S6E7_SEEDS="${SEED}" S6E7_RUN_TAG="_u_s${SEED}" \
    $PY src/train_realmlp_td.py 2>&1 \
    | tee "results/log_imp_s${SEED}.txt"
  if [ ! -f "results/oof_realmlp_te13_imp_u_s${SEED}.npy" ]; then
    echo "FATAL: seed ${SEED} did not produce oof_realmlp_te13_imp_u_s${SEED}.npy"
    echo "       (recipe drift — check S6E7_TE13/S6E7_IMP). Aborting lane."
    exit 1
  fi
  echo "=== seed ${SEED} done $(date -Is) ==="
done
echo "=== imp breadth lane complete $(date -Is) ==="
