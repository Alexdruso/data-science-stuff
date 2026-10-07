"""CatBoost leg for PS S6E10 — ordered boosting for diversity against LGBM/XGB.

    python src/train_catboost.py
    S6E10_CATBOOST_DEVICE=CPU python src/train_catboost.py

GPU settings follow the 6GB RTX 2060 recipe: `max_ctr_complexity=1`, no
`eval_set` (fixed iteration count, val predicted afterwards), a capped
`gpu_ram_part` (env-overridable; 0.5 leaves room for a concurrent GPU leg), and
`border_count=254` explicitly — the GPU default of 128 silently cost s6e8 most
of CatBoost's gap to LightGBM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET
from matrices import build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

_, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

DEVICE = os.environ.get("S6E10_CATBOOST_DEVICE", "GPU").upper()
GPU_RAM_PART = float(os.environ.get("S6E10_CATBOOST_GPU_RAM_PART", "0.5"))

PARAMS: dict[str, object] = {
    "iterations": 2000,
    "learning_rate": 0.06,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "max_ctr_complexity": 1,
    "border_count": 254,
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}


def _prepare(frame: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cat_cols:
        out[col] = out[col].astype(str)
    return out


def main() -> None:
    route = "--route" in sys.argv
    m = build_matrices()
    if route:
        from route import add_route_global
        add_route_global(m)
    params = load_params(RESULTS_DIR, PARAMS, "best_params_catboost.json")
    tuned = (RESULTS_DIR / "best_params_catboost.json").exists()
    device_params: dict[str, object] = {"task_type": DEVICE}
    if DEVICE == "GPU":
        device_params["gpu_ram_part"] = GPU_RAM_PART
    else:
        device_params["thread_count"] = 4
    print(f"CatBoost on {DEVICE}: train {m.X.shape}  test {m.X_test.shape}", flush=True)

    X = _prepare(m.X, m.cat_cols)
    X_test = _prepare(m.X_test, m.cat_cols)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        _y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if route:
            from route import route_te
            x_tr, x_va, x_te = route_te(x_tr, y_tr, [x_va, x_te])
        model = CatBoostClassifier(
            **params, **device_params, cat_features=m.cat_cols
        )
        model.fit(x_tr, y_tr)
        print(f"  fold {fold} fitted", flush=True)
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(
        X, m.y, X_test, fit_fold, n_outputs=1, score_fn=fold_score
    )
    for fold, score in enumerate(fold_scores, 1):
        print(f"  Fold {fold} AUC: {score:.5f}")
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"\nOOF AUC: {oof_auc:.5f}")

    run_name = "catboost_route" if route else ("catboost_v1_tuned" if tuned else "catboost_v1")
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test)
    out = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", m.test_ids, TARGET, test)
    print(f"Arrays → {RESULTS_DIR}   Submission → {out}")


if __name__ == "__main__":
    main()
