"""TabM (pytabkit) — parameter-efficient deep-ensemble MLP (zoo Z3, PS S6E7).

Gorishniy 2024; the strongest recent tabular-NN family never priced on this
dataset. Clean raw-feature run (median-impute + missingness indicators; cats as
strings with an explicit NA level) so the measurement is the architecture, not
the input recipe (Z1 carries that). pytabkit does its own internal val split
for early stopping; our val fold stays untouched (leakage-safe).

Speed (s6e6 notes): compile_model=True + allow_amp=True are mandatory on this
card; batch_size 2048 with sqrt-scaled lr. Timebox: if fold 1 exceeds ~25 min,
kill the run and record TabM as priced-out.

Run: S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_tabm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
LR = 0.004


def main() -> None:
    ds = load_dataset()
    num_cols = [c for c in ds.feature_cols if c not in ds.cat_cols]
    print(f"device {DEVICE}  batch {BATCH_SIZE}  lr {LR}")

    num_tr = ds.train[num_cols].to_numpy(dtype=np.float64)
    num_te = ds.test[num_cols].to_numpy(dtype=np.float64)
    miss_names = [f"miss_{c}" for c in num_cols + ds.cat_cols]

    def frame(num_filled: np.ndarray, src: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(num_filled, columns=num_cols)
        for c in ds.cat_cols:
            out[c] = src[c].fillna("NA").astype(str).to_numpy()
        miss = src[num_cols + ds.cat_cols].isna().to_numpy(dtype=np.float32)
        for i, n in enumerate(miss_names):
            out[n] = miss[:, i]
        return out

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        med = np.nanmedian(num_tr[tr_idx], axis=0)
        X_all = frame(np.where(np.isnan(num_tr), med, num_tr), ds.train)
        X_test = frame(np.where(np.isnan(num_te), med, num_te), ds.test)

        model = TabM_D_Classifier(
            device=DEVICE,
            random_state=fold_seed,
            val_metric_name="1-balanced_accuracy",
            batch_size=BATCH_SIZE,
            lr=LR,
            compile_model=True,
            allow_amp=True,
            verbosity=1,
        )
        model.fit(X_all.iloc[tr_idx], ds.y[tr_idx], cat_col_names=list(ds.cat_cols))
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        val_proba = model.predict_proba(X_all.iloc[val_idx]).astype(np.float64)
        test_proba = model.predict_proba(X_test).astype(np.float64)
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        test_proba /= test_proba.sum(axis=1, keepdims=True)

        del model, X_all, X_test
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"tabm_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("tabm", ds, oof, test, fold_scores)
    clear_ckpt(f"tabm_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
