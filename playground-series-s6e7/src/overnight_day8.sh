#!/usr/bin/env bash
# Day-8 overnight queue (2026-07-10). Everything skip-if-exists / fold-checkpointed,
# so a box reboot just needs this script rerun. GPU chain is strictly serial
# (6 GB card, one trainer at a time); CPU chain runs concurrently.
set -u
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
COMP_DIR="$(dirname "$SRC_DIR")"
cd "$COMP_DIR" || exit 1
PY="$(dirname "$COMP_DIR")/.venv/bin/python"
export PYTHONUNBUFFERED=1

gpu_chain() {
  # 1) _r2 rebuild remainder (mult 0.5; skip-if-exists resumes past finished seeds)
  S6E7_TAG_PREFIX=_r2 S6E7_REPAIR=1 S6E7_REPAIR_MULT=0.5 \
    bash src/run_breadth.sh xgboost catboost
  # 2) realmlp on the m050 surface, 3 seeds — completes the surface-consistent
  #    all-_r2 candidate (mult 0.5 + TE-via-realmlp + realmlp + best combiner)
  for seed in 42 7 123; do
    if [[ -f "results/oof_realmlp_r2_s${seed}.npy" ]]; then
      echo "skip realmlp_r2_s${seed} (exists)"; continue
    fi
    S6E7_REPAIR=1 S6E7_REPAIR_MULT=0.5 S6E7_SEEDS="$seed" S6E7_RUN_TAG="_r2_s${seed}" \
      "$PY" src/train_realmlp_td.py > "results/log_zoo_realmlp_r2_s${seed}.txt" 2>&1
  done
  # 3) zoo Z4 DANN + mask-consistency (trains on RAW matrix by design — no S6E7_REPAIR)
  S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 "$PY" src/train_dann.py \
    > results/log_zoo_dann.txt 2>&1
  # 4) zoo Z3 TabM retry (priced out of the day at >25 min/fold; free overnight)
  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 "$PY" src/train_tabm.py \
    > results/log_zoo_tabm.txt 2>&1
}

cpu_chain() {
  S6E7_TAG_PREFIX=_r2 S6E7_REPAIR=1 S6E7_REPAIR_MULT=0.5 \
    bash src/run_breadth.sh lgbm hgbc
}

gpu_chain > results/overnight_gpu.log 2>&1 &
cpu_chain > results/overnight_cpu.log 2>&1 &
wait
# Combine the _r2 per-seed arrays into breadth bases. NO cross-lineage blend here:
# realmlp_r_breadth lives on the m100 surface, the _r2 bases on m050 — mixing val
# surfaces breaks the compare-repaired-to-repaired rule. Tomorrow's call: either
# blend within _r2 only, or retrain realmlp seeds under mult 0.5 (_r2 tags) first.
S6E7_TAG_PREFIX=_r2 S6E7_REPAIR=1 S6E7_REPAIR_MULT=0.5 "$PY" src/combine_breadth.py \
  > results/log_combine_r2.txt 2>&1
S6E7_TAG_PREFIX=_r2 S6E7_REPAIR=1 S6E7_REPAIR_MULT=0.5 "$PY" -c "
import sys; sys.path.insert(0, 'src')
from combine_breadth import combine_seeds
from train_common import load_dataset
ds = load_dataset()
combine_seeds('realmlp', ds.y)
" > results/log_combine_realmlp_r2.txt 2>&1
echo "OVERNIGHT ALL-COMPLETE"
