"""Optuna hyperparameter search for the XGBoost leg.

    python src/tune_xgboost.py --arm s1 --trials 60 --timeout 14400

**Why XGBoost and not LightGBM.** The Day-10 breadth blend puts ~0.78 of its
weight on XGBoost and ~0.22 on the six-seed LightGBM bag; CatBoost and the MLP
both get exactly 0.0. Tuning effort should follow the weight. `PARAMS` in
`train_xgboost.py` are hand-set (lr 0.03 / depth 8 / 3000 trees) and have never
been searched.

Tuning is also the one lever a combiner cannot absorb. Root CLAUDE.md lesson #7
says a strong combiner eats leg-level gains — today's CatBoost repair was worth
+0.00272 solo and **+0.000000** to the blend, which is that lesson in one line.
But tuning *replaces* the leg rather than adding a member, so there is no
combiner above it to absorb the gain. That is why the pre-registration gives
tuning the lower +0.0005 bar.

Mirrors `tune_lgbm.py` deliberately:

* **3 folds, not 5** — hyperparameter *ranking* is a coarse decision; the winner
  is re-fit at 5 folds by `train_xgboost.py`.
* **sqlite-backed and resumable.** This box hard-cut power mid-run on 2026-08-06
  at 09:40:53 and lost a morning. `load_if_exists` makes the identical command a
  resume, so a crash costs exactly the running trial.
* **The untuned reference is enqueued as trial 0** on a fresh study, so every
  later trial is read against the incumbent rather than against nothing. On the
  LightGBM study that reference is *still* the best after six trials — a seeded
  reference is what makes that fact visible instead of invisible.

Device is CPU to match the leg it is tuning (`train_xgboost.py` uses
`tree_method="hist"` on CPU); a device switch would move the number by about the
size of the effect being measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from run_arm import ARMS, build_matrices

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

TUNE_FOLDS = 3
FIXED: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "enable_categorical": True,
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": 42,
}

#: The shipped `train_xgboost.py` point, seeded as trial 0 on a fresh study.
REFERENCE = {
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 1e-8,
    "gamma": 1e-8,
    "max_bin": 256,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1", choices=sorted(ARMS))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--out", default="best_params_xgboost.json")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    FIXED["n_jobs"] = args.n_jobs

    builder, _, description = ARMS[args.arm]
    x, y, _ = build_matrices(builder)
    print(f"tuning xgb on arm {args.arm} ({description}): {x.shape}", flush=True)

    folds = list(
        StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=42).split(x, y)
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        }
        oof = np.zeros(len(y))
        for tr_idx, va_idx in folds:
            model = XGBClassifier(**params)
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(x.iloc[va_idx], y[va_idx])],
                verbose=False,
            )
            oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
            trial.set_user_attr("best_iteration", model.best_iteration)
            del model
        return float(roc_auc_score(y, oof))

    db = RESULTS_DIR / f"optuna_xgb_{args.arm}.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{db}",
        study_name=f"xgb_{args.arm}",
        load_if_exists=True,
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        print(
            f"resuming study 'xgb_{args.arm}' from {db.name}: "
            f"{len(done)} completed trials, best so far {study.best_value:.5f}",
            flush=True,
        )
    if not study.trials:
        study.enqueue_trial(dict(REFERENCE))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  trial {trial.number:>3}  {trial.value:.5f}   best {study.best_value:.5f}",
            flush=True,
        )

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=[log])

    best = {**FIXED, **study.best_params}
    best["n_estimators"] = 3000
    out = RESULTS_DIR / args.out
    out.write_text(json.dumps(best, indent=2, sort_keys=True))
    print(f"\nbest {TUNE_FOLDS}-fold OOF AUC: {study.best_value:.5f}")
    print(f"best params -> {out}")
    print(json.dumps(study.best_params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
