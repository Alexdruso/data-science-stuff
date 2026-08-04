"""Generic CV runner for a non-LightGBM leg on a chosen feature surface.

`run_arm.py` owns the LightGBM arm sweep; this owns everything else. Both go
through the same `ARMS` table and the same fold seeds, so every OOF array lands
row-aligned and stackable — provided the arm (the *surface*) matches. Arrays
from different arms must never be blended together (root CLAUDE.md lesson #9).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from run_arm import ARMS, SCORES_PATH, assert_budget_constraint, build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

N_FOLDS = 5

FitModel = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, int],
    tuple[np.ndarray, np.ndarray],
]


def run_leg(name: str, arm: str, seed: int, fit_model: FitModel) -> float:
    """Fit one leg on one surface at one seed; save OOF/test arrays; return OOF AUC."""
    builder, transform, _ = ARMS[arm]
    x, y, x_test = build_matrices(builder)
    assert_budget_constraint(x)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if transform is not None:
            # Fold-aware: every statistic is fit on the training split only.
            x_tr, x_va, x_te = transform(x_tr, y_tr, x_va, x_te)
        return fit_model(x_tr, y_tr, x_va, y_va, x_te, fold)

    oof, test_pred, fold_scores = run_cv(
        x,
        y,
        x_test,
        fit_fold,
        n_splits=N_FOLDS,
        seed=seed,
        n_outputs=1,
        score_fn=lambda yv, p: float(roc_auc_score(yv, p)),
    )
    auc = float(roc_auc_score(y, oof))
    tag = f"{name}_{arm}_s{seed}"
    np.save(RESULTS_DIR / f"oof_{tag}.npy", oof)
    np.save(RESULTS_DIR / f"test_{tag}.npy", test_pred)
    save_cv_result(RESULTS_DIR, tag, fold_scores, auc, metric_name="auc")

    scores: dict[str, dict[str, float]] = {}
    if SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text())
    scores.setdefault(f"{name}_{arm}", {})[str(seed)] = auc
    SCORES_PATH.write_text(json.dumps(scores, indent=2, sort_keys=True))

    print(f"  {tag}: features={x.shape[1]:>3}  OOF AUC {auc:.5f}", flush=True)
    return auc
