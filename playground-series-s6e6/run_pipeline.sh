#!/bin/bash
# Run full pipeline after Optuna finishes.
# Usage: ./run_pipeline.sh [--skip-lgbm] [--skip-xgb] [--skip-cb]
#
# Expects best_params.json to already exist (from tune_lgbm.py).
# XGBoost and CatBoost training scripts can also be run independently.

set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate

SKIP_LGBM=0
SKIP_XGB=0
SKIP_CB=0

for arg in "$@"; do
  case "$arg" in
    --skip-lgbm) SKIP_LGBM=1 ;;
    --skip-xgb)  SKIP_XGB=1  ;;
    --skip-cb)   SKIP_CB=1   ;;
  esac
done

if [ $SKIP_LGBM -eq 0 ]; then
  echo "=== Running LGBM with tuned params ==="
  python src/baseline.py
fi

if [ $SKIP_XGB -eq 0 ]; then
  echo "=== Running XGBoost ==="
  python src/train_xgboost.py
fi

if [ $SKIP_CB -eq 0 ]; then
  echo "=== Running CatBoost ==="
  python src/train_catboost.py
fi

echo "=== Running ensemble ==="
python src/ensemble.py
