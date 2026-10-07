"""PR-14 — re-tune LGBM on the route + residual feature set.

    python src/tune_lgbm_route.py            # search → results/best_params_lgbm_pr14.json
    S6E10_LGBM_PARAMS=best_params_lgbm_pr14.json python src/tune_lgbm.py --evaluate --route --resid

Same protocol as PR-5 (3-fold on a fixed 50% subsample, seed 0), except the
route TE + residual columns are fit inside each search fold ONCE and cached, so
every trial is a pure LGBM fit. Trial 0 is the PR-5 config (the incumbent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from matrices import build_matrices
from route import add_route_global, route_te
from run_lgbm_arm import route_residual

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
OUT = RESULTS_DIR / "best_params_lgbm_pr14.json"
N_TRIALS = 20


def main() -> None:
    m = build_matrices()
    add_route_global(m)
    idx = np.random.default_rng(0).permutation(len(m.y))[: len(m.y) // 2]
    X, y = m.X.iloc[idx].reset_index(drop=True), m.y[idx]
    folds = []
    for tr, va in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y):
        x_tr, x_va = route_te(X.iloc[tr], y[tr], [X.iloc[va]])
        x_tr, x_va = route_residual(x_tr, y[tr], [x_va])
        folds.append((x_tr, y[tr], x_va, y[va], va))
    print(f"search (CPU): {X.shape} → {folds[0][0].shape[1]} feats", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary", "metric": "auc", "n_estimators": 6000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1, "random_state": 42, "verbose": -1, "n_jobs": 10,
        }
        oof = np.zeros(len(y))
        iters = []
        for x_tr, y_tr, x_va, y_va, va in folds:
            model = LGBMClassifier(**params).fit(
                x_tr, y_tr, eval_set=[(x_va, y_va)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            oof[va] = model.predict_proba(x_va)[:, 1]
            iters.append(model.best_iteration_)
        trial.set_user_attr("best_iters", iters)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    pr5 = json.loads((RESULTS_DIR / "best_params_lgbm_pr5.json").read_text())
    study.enqueue_trial({k: pr5[k] for k in (
        "learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "colsample_bytree", "subsample")})
    def save_best(s: optuna.Study, _t: optuna.trial.FrozenTrial) -> None:
        OUT.write_text(json.dumps({**s.best_params, "n_estimators": 6000, "subsample_freq": 1}, indent=2))

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_best,
        lambda s, t: print(
            f"  trial {t.number:>2}: {t.value:.5f}  best {s.best_value:.5f}  "
            f"iters {t.user_attrs.get('best_iters')}  {t.params}", flush=True)
    ])
    OUT.write_text(json.dumps({**study.best_params, "n_estimators": 6000, "subsample_freq": 1}, indent=2))
    print(f"\nbest {study.best_value:.5f} (incumbent trial {study.trials[0].value:.5f}) → {OUT}")


if __name__ == "__main__":
    main()
