"""PR-5: Optuna tuning of the LGBM leg, then a paired 5-fold read.

    python src/tune_lgbm.py --search      # 40 TPE trials → results/best_params_lgbm_pr5.json
    python src/tune_lgbm.py --evaluate    # 5-fold with the best params, paired vs control

Search: 3-fold AUC on a fixed 50% subsample (seed 0), on whatever device
`get_lgbm_device()` finds (CUDA here) — device only affects wall clock of the
search. Evaluate: the PR-1 control's exact CPU / 8-thread / seed-42 config with
the tuned hyperparameters swapped in, so the per-fold delta is paired.
Space and bar are pre-registered in CLAUDE.md (PR-5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from lgbm_device import get_lgbm_device
from matrices import build_matrices
from train_lgbm_orig import PARAMS as CONTROL_PARAMS

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
PARAMS_FILE = RESULTS_DIR / "best_params_lgbm_pr5.json"
CONTROL = "lgbm_pr1_base"
N_TRIALS = 40
BAR = 0.0005


def search() -> None:
    m = build_matrices()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(m.y))[: len(m.y) // 2]
    X, y = m.X.iloc[idx].reset_index(drop=True), m.y[idx]
    device, n_jobs = get_lgbm_device()
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
    print(f"search on {device}: {X.shape}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "random_state": 42,
            "verbose": -1,
            "device_type": device,
            "n_jobs": n_jobs,
        }
        oof = np.zeros(len(y))
        iters = []
        for tr, va in folds:
            model = LGBMClassifier(**params)
            model.fit(
                X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            iters.append(model.best_iteration_)
        trial.set_user_attr("best_iters", iters)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    # Seed the search with the control config so "tuned" is never worse than it on the subsample.
    study.enqueue_trial({
        "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 20,
        "reg_lambda": 1e-3, "colsample_bytree": 0.8, "subsample": 0.8,
    })
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[
        lambda s, t: print(
            f"  trial {t.number:>2}: {t.value:.5f}  best {s.best_value:.5f}  "
            f"iters {t.user_attrs.get('best_iters')}  {t.params}", flush=True)
    ])
    best = {**study.best_params, "n_estimators": 5000, "subsample_freq": 1}
    PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nbest {study.best_value:.5f} (control trial {study.trials[0].value:.5f}) → {PARAMS_FILE}")


def evaluate(route: bool = False, resid: bool = False, orig: bool = False) -> None:
    import os
    params_file = RESULTS_DIR / os.environ.get("S6E10_LGBM_PARAMS", PARAMS_FILE.name)
    tuned = json.loads(params_file.read_text())
    nfolds = int(os.environ.get("S6E10_NFOLDS", "5"))
    params = {**CONTROL_PARAMS, **tuned}
    print(f"evaluate (CPU, paired): {tuned}", flush=True)
    m = build_matrices(with_original=orig)
    n_test = len(m.X_test)
    if orig:
        m.X_test = pd.concat([m.X_test, m.X_orig], ignore_index=True)
    if route:
        from route import add_route_global
        add_route_global(m)
    x_orig = None
    if orig:
        x_orig = m.X_test.iloc[n_test:].reset_index(drop=True)
        m.X_test = m.X_test.iloc[:n_test].reset_index(drop=True)
    extras = [e for e in os.environ.get("S6E10_EXTRAS", "").split(",") if e]
    if extras:
        from route import add_extras
        add_extras(m, x_orig, extras)
        print(f"extras {extras}: train {m.X.shape}", flush=True)

    def fit_fold(
        x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray,
        x_te: pd.DataFrame, fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if route:
            from route import route_te
            if x_orig is not None:
                x_tr, x_va, x_te, x_o = route_te(x_tr, y_tr, [x_va, x_te, x_orig])
                x_tr = pd.concat([x_tr, x_o[x_tr.columns]], ignore_index=True)
                y_tr = np.concatenate([y_tr, m.y_orig])
            else:
                x_tr, x_va, x_te = route_te(x_tr, y_tr, [x_va, x_te])
        if resid:
            from run_lgbm_arm import route_residual
            x_tr, x_va, x_te = route_residual(x_tr, y_tr, [x_va, x_te])
        model = LGBMClassifier(**params)
        model.fit(
            x_tr, y_tr, eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        val = model.predict_proba(x_va)[:, 1]
        print(f"  fold {fold}: best_iter={model.best_iteration_} AUC {roc_auc_score(y_va, val):.5f}", flush=True)
        return val, model.predict_proba(x_te)[:, 1]

    oof, test, fold_scores = run_cv(
        m.X, m.y, m.X_test, fit_fold, n_outputs=1, n_splits=nfolds,
        score_fn=lambda yv, pv: float(roc_auc_score(yv, pv)),
    )
    oof_auc = float(roc_auc_score(m.y, oof))
    name = ("lgbm_pr13_resid" if resid else "lgbm_pr12_route") if route else "lgbm_pr5_tuned"
    name += "".join(f"_{e}" for e in extras)
    name += os.environ.get("S6E10_LGBM_SUFFIX", "") + ("_orig" if orig else "")
    name += "" if nfolds == 5 else f"_k{nfolds}"
    save_cv_result(RESULTS_DIR, name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{name}.npy", test)

    ctrl = np.load(RESULTS_DIR / f"oof_{CONTROL}.npy")
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
    deltas = [roc_auc_score(m.y[va], oof[va]) - roc_auc_score(m.y[va], ctrl[va]) for _, va in splits]
    d_oof = oof_auc - float(roc_auc_score(m.y, ctrl))
    n_pos = sum(d > 0 for d in deltas)
    verdict = "PASS" if d_oof >= BAR and n_pos >= 4 else "FAIL"
    print(f"\n{name}: OOF {oof_auc:.6f}  per-fold delta {np.round(deltas, 6).tolist()}")
    print(f"{name}: OOF delta {d_oof:+.6f}  positive folds {n_pos}/5  → {verdict}")
    if route:
        tuned_oof = np.load(RESULTS_DIR / ("oof_lgbm_pr12_route.npy" if resid else "oof_lgbm_pr5_tuned.npy"))
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
        td = [roc_auc_score(m.y[va], oof[va]) - roc_auc_score(m.y[va], tuned_oof[va]) for _, va in splits]
        dt = oof_auc - float(roc_auc_score(m.y, tuned_oof))
        print(f"vs {'lgbm_pr12_route' if resid else 'lgbm_pr5_tuned'}: OOF delta {dt:+.6f}  per-fold {np.round(td, 6).tolist()}  "
              f"→ {'PASS' if dt >= BAR and sum(d > 0 for d in td) >= 4 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--route", action="store_true", help="PR-12 route features")
    parser.add_argument("--resid", action="store_true", help="PR-13 residual route encoding (with --route)")
    parser.add_argument("--orig", action="store_true", help="append original rows to each training fold (with --route)")
    args = parser.parse_args()
    if args.search:
        search()
    if args.evaluate:
        evaluate(route=args.route, resid=args.resid, orig=args.orig)


if __name__ == "__main__":
    main()
