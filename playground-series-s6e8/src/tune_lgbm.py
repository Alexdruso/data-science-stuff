"""Optuna hyperparameter search for the LightGBM leg.

    python src/tune_lgbm.py --arm s0 --trials 60 --timeout 3600

The reference leg is untuned (`num_leaves=127`, `lr=0.05`, no search), so this
has a better prior than any of the feature levers in the sweep.

Tuning uses **3 folds**, not 5: hyperparameter *ranking* is a coarse decision
and 3 folds resolves it at roughly half the cost, which buys more trials. The
winning params are then re-fit at 5 folds by `run_arm.py` / `baseline.py`.

Writes `results/best_params.json`, which `baseline.py` already picks up
automatically through `load_params`. Device is locked to CPU to match the sweep
(see `run_arm.py` — the cuda/cpu gap on this box is 0.0003 AUC, the same size
as the effects being measured).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from run_arm import ARMS, build_matrices

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

TUNE_FOLDS = 3
FIXED: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 3000,
    "verbose": -1,
    "n_jobs": 12,
    "device_type": "cpu",
    "random_state": 42,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s0", choices=sorted(ARMS))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out", default="best_params.json")
    args = parser.parse_args()

    builder, _, description = ARMS[args.arm]
    x, y, _ = build_matrices(builder)
    print(f"tuning on arm {args.arm} ({description}): {x.shape}", flush=True)

    folds = list(
        StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=42).split(x, y)
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
        }
        oof = np.zeros(len(y))
        for tr_idx, va_idx in folds:
            model = LGBMClassifier(**params)
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(x.iloc[va_idx], y[va_idx])],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
            trial.set_user_attr("best_iteration", model.best_iteration_)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.enqueue_trial(
        {
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 20,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_alpha": 1e-8,
            "reg_lambda": 1e-8,
            "max_bin": 255,
        }
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  trial {trial.number:>3}  {trial.value:.5f}   best {study.best_value:.5f}",
            flush=True,
        )

    study.optimize(
        objective, n_trials=args.trials, timeout=args.timeout, callbacks=[log]
    )

    best = {**FIXED, **study.best_params}
    best["n_estimators"] = 3000
    out = RESULTS_DIR / args.out
    out.write_text(json.dumps(best, indent=2, sort_keys=True))
    print(f"\nbest {TUNE_FOLDS}-fold OOF AUC: {study.best_value:.5f}")
    print(f"best params -> {out}")
    print(json.dumps(study.best_params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
