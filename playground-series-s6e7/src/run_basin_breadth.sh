#!/usr/bin/env bash
# Day-21 hail-Mary escalation lane (CPU): bring the HGBC TPE-basin arm
# (`hgbc_te3_u_s42t`) to BREADTH PARITY with the deployed 8-seed
# hgbc_te3_u_breadth.  Seed 42 already exists; this runs the other seven.
#
# Chained AFTER probe_passthrough so the two don't contend for the same cores
# (Ryzen 1600, 6c/12t; the probe already runs n_jobs=4 and the GPU lane's
# dataloader wants threads too).
#
# u4's lineage is never touched: S6E7_HGBC_PARAMS hard-requires an explicit
# fresh S6E7_RUN_TAG.  Reboot-safe via per-fold checkpoints + skip-if-exists.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python

# TPE basin, verbatim from results/log_hgbc_te3_basin.txt
export S6E7_HGBC_PARAMS='{"learning_rate": 0.03373693124435465, "max_iter": 600, "max_leaf_nodes": 77, "min_samples_leaf": 391, "l2_regularization": 0.09110510408304606, "max_features": 0.40895869469862856, "max_bins": 192, "max_depth": null}'
export PYTHONUNBUFFERED=1

for SEED in 7 123 2024 99 2025 31 7777; do
  if [ -f "results/oof_hgbc_te3_u_s${SEED}t.npy" ]; then
    echo "=== seed ${SEED}: already present, skipping ==="
    continue
  fi
  echo "=== seed ${SEED} starting $(date -Is) ==="
  S6E7_SEEDS="${SEED}" S6E7_RUN_TAG="_u_s${SEED}t" \
    $PY src/train_hgbc_te3.py 2>&1 \
    | tee "results/log_basin_s${SEED}.txt"
  echo "=== seed ${SEED} done $(date -Is) ==="
done
echo "=== basin breadth lane complete $(date -Is) ==="
