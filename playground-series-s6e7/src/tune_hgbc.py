"""Optuna HP sweep for hgbc with the DEPLOYED objective (plan #3).

The 07-06 kill-audit un-killed HP tuning: zero trials were ever run. hgbc goes
first because it fits in ~5 s/fold on CPU; LGBM/XGB/Cat get a sweep only if hgbc
shows >0.001 headroom over its default config (else tuning is dead for all).

Objective = single-seed (42) 5-fold OOF **decision-weighted** balanced accuracy —
what we deploy — not logloss and not plain argmax. Weight tuning inside the
objective uses n_restarts=3 (cheap, identical treatment for every trial, so
rankings are fair); the winner gets a full-fat refit outside the study.

The study lives in results/optuna_hgbc.db (sqlite) so a hard reboot resumes
where it left off; the default HGBC_PARAMS config is enqueued as trial 0 to pin
the incumbent baseline. Per-fold argmax bacc is reported for median pruning.

Run: ../.venv/bin/python src/tune_hgbc.py | tee -a results/tune_hgbc.txt
Env: S6E7_TUNE_SECONDS (default 3600).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SEED = 42

FIXED: dict[str, object] = {
    "loss": "log_loss",
    "scoring": "balanced_accuracy",
    "class_weight": "balanced",
    "max_iter": 10000,
    "n_iter_no_change": 10,
    "validation_fraction": 0.1,
    "categorical_features": "from_dtype",
    "random_state": SEED,
}

# train_hgbc.py incumbent (weighted OOF 0.9494 bagged; enqueued as trial 0)
DEFAULT_TRIAL: dict[str, object] = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "max_depth": 0,  # 0 encodes None
    "min_samples_leaf": 20,
    "l2_regularization": 1e-3,
    "max_bins": 255,
}


def suggest(trial: optuna.Trial) -> dict[str, object]:
    depth = trial.suggest_int("max_depth", 0, 12)  # 0 -> None (unbounded)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "max_depth": None if depth == 0 else depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-3, 10.0, log=True
        ),
        "max_bins": trial.suggest_categorical("max_bins", [128, 192, 255]),
    }


def main() -> None:
    ds = load_dataset()
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    y = ds.y
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(y, y))

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        oof = np.zeros((len(y), N_CLASSES))
        fold_scores = []
        for i, (tr_idx, val_idx) in enumerate(folds):
            model = HistGradientBoostingClassifier(**FIXED, **params)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            assert list(model.classes_) == list(range(N_CLASSES))
            oof[val_idx] = model.predict_proba(X.iloc[val_idx])
            fold_scores.append(
                balanced_accuracy_score(y[val_idx], oof[val_idx].argmax(1))
            )
            trial.report(float(np.mean(fold_scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned
        w = tune_decision_weights(y, oof, n_restarts=3)
        return float(balanced_accuracy_score(y, weighted_predict(oof, w)))

    study = optuna.create_study(
        study_name="hgbc",
        storage=f"sqlite:///{RESULTS_DIR / 'optuna_hgbc.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    if not study.trials:  # pin the incumbent as trial 0 on first run only
        study.enqueue_trial(DEFAULT_TRIAL)

    timeout = int(os.environ.get("S6E7_TUNE_SECONDS", "3600"))
    study.optimize(objective, timeout=timeout, gc_after_trial=True)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{len(done)} complete trials ({len(study.trials)} total)")
    incumbent = next((t for t in done if t.number == 0), None)
    base = incumbent.value if incumbent else float("nan")
    print(f"incumbent (default params): {base:.4f}")
    print(f"best: {study.best_value:.4f} (trial {study.best_trial.number})")
    print(f"best params: {study.best_params}")
    delta = study.best_value - base
    print(f"headroom vs incumbent: {delta:+.4f} -> "
          f"{'PASS (sweep LGBM/XGB/Cat next)' if delta >= 0.001 else 'FAIL (tuning dead for the family)'}")


if __name__ == "__main__":
    main()
