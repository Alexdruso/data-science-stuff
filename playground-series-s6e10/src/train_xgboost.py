"""XGBoost leg for PS S6E10 — diversity against the LightGBM baseline.

    python src/train_xgboost.py

GPU (`device="cuda"`) by default; `S6E10_XGB_DEVICE=cpu` to override.
Categoricals go in as pandas `category` with `enable_categorical=True`;
`build_matrices` has unioned the level sets so codes match across splits.
Params are the s6e8 hand-set XGB config (same task shape, similar size).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET
from matrices import build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

_, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

DEVICE = os.environ.get("S6E10_XGB_DEVICE", "cuda")
PARAMS: dict[str, object] = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "enable_categorical": True,
    "tree_method": "hist",
    "device": DEVICE,
    "n_jobs": 2,
    "random_state": 42,
    "verbosity": 0,
}


def main() -> None:
    route = "--route" in sys.argv
    orig = "--orig" in sys.argv  # original rows appended to each training fold (with --route)
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
    params = load_params(RESULTS_DIR, PARAMS, "best_params_xgboost.json")
    tuned = (RESULTS_DIR / "best_params_xgboost.json").exists()
    print(f"XGBoost on {DEVICE}: train {m.X.shape}  test {m.X_test.shape}", flush=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if route:
            from route import route_te
            if x_orig is not None:
                x_tr, x_va, x_te, x_o = route_te(x_tr, y_tr, [x_va, x_te, x_orig])
                x_tr = pd.concat([x_tr, x_o[x_tr.columns]], ignore_index=True)
                y_tr = np.concatenate([y_tr, m.y_orig])
            else:
                x_tr, x_va, x_te = route_te(x_tr, y_tr, [x_va, x_te])
        model = XGBClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        print(f"  fold {fold} best_iter={model.best_iteration}", flush=True)
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(
        m.X, m.y, m.X_test, fit_fold, n_outputs=1, score_fn=fold_score
    )
    for fold, score in enumerate(fold_scores, 1):
        print(f"  Fold {fold} AUC: {score:.5f}")
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"\nOOF AUC: {oof_auc:.5f}")

    run_name = "xgb_route" if route else ("xgb_v1_tuned" if tuned else "xgb_v1")
    run_name += "".join(f"_{e}" for e in extras) + ("_orig" if orig else "")
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test)
    out = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", m.test_ids, TARGET, test)
    print(f"Arrays → {RESULTS_DIR}   Submission → {out}")


if __name__ == "__main__":
    main()
