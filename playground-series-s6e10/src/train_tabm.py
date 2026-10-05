"""TabM leg for PS S6E10 (PR-8) — parameter-efficient deep-ensemble MLP (pytabkit).

    python src/train_tabm.py

Settings follow s6e7's `train_tabm.py` (6GB RTX 2060): `compile_model=True` and
`allow_amp=True` are mandatory for speed; batch 2048 with lr 0.004. pytabkit
early-stops on its own internal split of the training fold, selecting by AUC
(`1-auc_ovr`), so our validation fold is never seen during fitting.

Numeric NaNs (only the arrival delay and its derived column) are filled with
fold-train medians; categoricals go in as strings via `cat_col_names`.
GPU memory rule: model deleted and allocator cleared inside every fold.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET
from matrices import build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission

_, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 2048
LR = 0.004


def to_frames(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], cat_cols: list[str]
) -> list[pd.DataFrame]:
    num_cols = [c for c in frames[0].columns if c not in cat_cols]
    medians = frames[0][num_cols].median().fillna(0.0)
    out = []
    for frame in frames:
        block = frame[num_cols].astype(np.float64).fillna(medians)
        for col in cat_cols:
            block[col] = frame[col].astype(str).to_numpy()
        out.append(block)
    return out


def main() -> None:
    m = build_matrices()
    X = m.X.drop(columns=["is_original"])
    X_test = m.X_test.drop(columns=["is_original"])
    print(f"TabM on {DEVICE}: train {X.shape} batch {BATCH_SIZE} lr {LR}", flush=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        tr, va, te = to_frames((x_tr, x_va, x_te), m.cat_cols)
        model = TabM_D_Classifier(
            device=DEVICE,
            random_state=SEED + fold * 100,
            val_metric_name="1-auc_ovr",
            batch_size=BATCH_SIZE,
            lr=LR,
            compile_model=True,
            allow_amp=True,
            verbosity=1,
        )
        model.fit(tr, y_tr, cat_col_names=list(m.cat_cols))
        assert list(model.classes_) == [0, 1], model.classes_
        val = model.predict_proba(va)[:, 1].astype(np.float64)
        test = model.predict_proba(te)[:, 1].astype(np.float64)
        print(
            f"  fold {fold}: AUC {roc_auc_score(y_va, val):.5f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del model, tr, va, te
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return val, test

    oof, test, fold_scores = run_cv(
        X, m.y, X_test, fit_fold, n_outputs=1,
        score_fn=lambda yv, pv: float(roc_auc_score(yv, pv)),
    )
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"\nfolds {[round(s, 5) for s in fold_scores]}  OOF AUC: {oof_auc:.5f}")
    run_name = "tabm_v1"
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test)
    out = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", m.test_ids, TARGET, test)
    print(f"Arrays → {RESULTS_DIR}   Submission → {out}")


if __name__ == "__main__":
    main()
