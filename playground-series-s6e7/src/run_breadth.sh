#!/usr/bin/env bash
# Track A — breadth-averaged 4-GBDT bagging (final #2), reboot-resumable.
#
# Runs each (model, seed) as its OWN process writing oof_<model>_s<seed>.npy via the
# existing trainers (S6E7_SEEDS=<seed> S6E7_RUN_TAG=_s<seed>). Skip-if-exists means a
# silent reboot (this box hard-reboots even at idle) only loses the in-flight fit; rerun
# the script and it resumes. Water is KEPT (features.py default) — champion recipe.
# combine_breadth.py then averages the surviving per-seed arrays into oof_<model>_bag.npy.
set -u
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
COMP_DIR="$(dirname "$SRC_DIR")"          # playground-series-s6e7
REPO_ROOT="$(dirname "$COMP_DIR")"        # data-science-stuff (holds .venv)
cd "$COMP_DIR" || exit 1
# Use the venv interpreter directly — `source activate` doesn't reliably take in a
# detached nohup shell, and the system has only python3, so bare `python` fails.
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY"; exit 1; }

# S6E7_TAG_PREFIX namespaces a variant run's artifacts (e.g. _r for the repaired
# recipe -> oof_lgbm_r_s42.npy); required because skip-if-exists would otherwise
# resume a repaired run from unrepaired arrays. S6E7_REPAIR passes through the
# environment to the trainers untouched.
TAG_PREFIX="${S6E7_TAG_PREFIX:-}"

SEEDS=(42 7 123 2024 99 2025 31 7777)
declare -A SCRIPT=( [lgbm]=src/baseline.py [hgbc]=src/train_hgbc.py \
                    [xgboost]=src/train_xgboost.py [catboost]=src/train_catboost.py )
# Model list from args (so GPU models can run separately from the CPU ones to avoid
# 6GB GPU contention with the NN); default = all four, CPU first.
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=(lgbm hgbc xgboost catboost); fi

export PYTHONUNBUFFERED=1
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="${TAG_PREFIX}_s${seed}"
    out="results/oof_${model}${tag}.npy"
    if [[ -f "$out" ]]; then
      echo "BREADTH skip ${model}${tag} (exists)"
      continue
    fi
    echo "BREADTH start ${model}${tag}"
    if S6E7_SEEDS="$seed" S6E7_RUN_TAG="$tag" "$PY" "${SCRIPT[$model]}" \
         > "results/log_breadth_${model}${tag}.txt" 2>&1; then
      echo "BREADTH done ${model}${tag} -> ${out}"
    else
      echo "BREADTH FAIL ${model}${tag} (see results/log_breadth_${model}${tag}.txt)"
    fi
  done
done
echo "BREADTH ALL-COMPLETE"
