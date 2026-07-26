#!/usr/bin/env bash
# Day-21 hail-Mary escalation lane (GPU): bring the realmlp joint-basin x IMP arm
# (`realmlp_te13_imp_u_s42ji`) to BREADTH PARITY with the deployed 8-seed
# realmlp_te13_u_breadth.  CANARY STAGE: seeds 7,123,2024 only (matched N=4 with the
# free stock-breadth-4), per the pre-registered gate in CLAUDE.md.
#
# Speculative: launched before probe_passthrough's gate reads, because the GPU is
# idle and this is the long pole (~35-40 min/seed).  If the gate kills the pitch
# these arrays are simply inventory — u4's lineage is never touched (every run
# carries a fresh explicit S6E7_RUN_TAG, which the trainer hard-requires whenever
# S6E7_REALMLP_PARAMS is set).
#
# Reboot-safe: per-fold checkpoints + skip-if-exists, per project_pc_random_reboots.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python

# joint-TPE basin (tune_realmlp_joint trial 59), verbatim from
# results/log_day17_realmlp_compose.txt
export S6E7_REALMLP_PARAMS='{"lr": 0.026838692514352534, "epochs": 9, "weight_decay": 0.06843739349150744, "dropout": 0.23790569176498397, "hidden_dims": [768, 768, 768], "loss_prior_power": 0.90737414967966, "ls_eps": 0.0051568034184429476, "numeric_noise_std": 0.020951523074649925}'
export S6E7_IMP=1
# ⚠️ S6E7_TE13=1 IS LOAD-BEARING. Omitting it silently trains a DIFFERENT model
# (MODEL_NAME drops the "_te13" segment -> realmlp_imp, ~0.9503 vs 0.95065) and
# writes it under a different key, so nothing errors — the 2026-07-26 first
# attempt lost 50 min to exactly this. The assert below makes it loud.
export S6E7_TE13=1
export PYTHONUNBUFFERED=1

# Single-instance lock: the first attempt was launched twice by accident and two
# runs raced on the same array files (identical output, but only by determinism).
LOCK=results/.ji_breadth.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_ji_breadth.sh holds $LOCK — refusing to start a second instance"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

for SEED in 7 123 2024; do
  if [ -f "results/oof_realmlp_te13_imp_u_s${SEED}ji.npy" ]; then
    echo "=== seed ${SEED}: already present, skipping ==="
    continue
  fi
  echo "=== seed ${SEED} starting $(date -Is) ==="
  S6E7_SEEDS="${SEED}" S6E7_RUN_TAG="_u_s${SEED}ji" \
    $PY src/train_realmlp_td.py 2>&1 \
    | tee "results/log_ji_s${SEED}.txt"
  # fail loudly if the recipe drifted: the array MUST land under the te13+imp key
  if [ ! -f "results/oof_realmlp_te13_imp_u_s${SEED}ji.npy" ]; then
    echo "FATAL: seed ${SEED} did not produce oof_realmlp_te13_imp_u_s${SEED}ji.npy"
    echo "       (recipe drift — check S6E7_TE13/S6E7_IMP). Aborting lane."
    exit 1
  fi
  echo "=== seed ${SEED} done $(date -Is) ==="
done
echo "=== ji breadth lane complete $(date -Is) ==="
