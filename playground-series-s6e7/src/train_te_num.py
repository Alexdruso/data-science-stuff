"""LGBM leg with exact-value target encoding of the numerics (notebook-sourced).

Source: Mark Susol's public S6E7 trail — +0.0009 OOF with matching LB, no
haircut, from target-encoding the numeric features BY THEIR EXACT (quantized)
VALUE, cross-fitted. Mechanism: the numerics are coarse grids (water 400
uniques); with 127 leaves a tree cannot isolate per-grid-point posteriors, so a
cross-fitted P(class | exact value) hands it resolution it cannot reach. This is
the target-mean sibling of train_freq.py's counts (+0.0001) and is the last
representation-recovery idea with a public verified effect size.

Fold safety (repo invariant #3): the encoding is target-derived, so TRAINING
rows are encoded via an INNER 5-fold cross-fit (each row's TE never saw its own
label), while val/test rows use maps fit on the full training fold. Smoothing
toward the fold's class prior (pseudo-count 20). NaN inputs stay NaN (the
missingness channel is untouched); observed-but-unseen values get the prior.

Run paired with lgbm_r_s42 (0.9478), repaired surface:
  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 \
      ../.venv/bin/python src/train_te_num.py
Gate: weighted OOF delta > +0.001 solo, or blend-level at the same margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from features import NUM_COLS
from train_common import N_CLASSES, Dataset, bagged_cv, finalize, load_dataset

SMOOTH = 20.0


def te_maps(
    x: pd.Series, y: np.ndarray, prior: np.ndarray
) -> list["pd.Series[float]"]:
    """Per-class smoothed posteriors keyed by exact value (NaN rows excluded)."""
    obs = x.notna().to_numpy()
    ct = pd.crosstab(x[obs], pd.Series(y[obs], index=x.index[obs]))
    ct = ct.reindex(columns=range(N_CLASSES), fill_value=0)
    n = ct.sum(axis=1)
    return [(ct[k] + SMOOTH * prior[k]) / (n + SMOOTH) for k in range(N_CLASSES)]


def apply_maps(
    x: pd.Series, maps: list["pd.Series[float]"], prior: np.ndarray
) -> dict[str, np.ndarray]:
    out = {}
    observed = x.notna()
    for k in range(N_CLASSES):
        m = x.map(maps[k])
        m[observed & m.isna()] = prior[k]  # seen in val/test, unseen in fold-train
        out[str(k)] = m.to_numpy(dtype="float32")
    return out


def encode_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    num_cols: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list["pd.Series[float]"]], np.ndarray]:
    """TE columns for the training fold (inner cross-fit) + full-fold maps."""
    y_tr = y[tr_idx]
    prior = np.bincount(y_tr, minlength=N_CLASSES) / len(y_tr)
    tr_cols = {
        f"te_{c}_{k}": np.full(len(tr_idx), np.nan, dtype="float32")
        for c in num_cols
        for k in range(N_CLASSES)
    }
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    for in_fit, in_map in inner.split(tr_idx):
        fit_idx, map_idx = tr_idx[in_fit], tr_idx[in_map]
        for c in num_cols:
            maps = te_maps(X[c].iloc[fit_idx], y[fit_idx], prior)
            enc = apply_maps(X[c].iloc[map_idx], maps, prior)
            for k in range(N_CLASSES):
                tr_cols[f"te_{c}_{k}"][in_map] = enc[str(k)]
    full_maps = {c: te_maps(X[c].iloc[tr_idx], y_tr, prior) for c in num_cols}
    return pd.DataFrame(tr_cols, index=X.index[tr_idx]), full_maps, prior


def with_te(
    X: pd.DataFrame,
    full_maps: dict[str, list["pd.Series[float]"]],
    prior: np.ndarray,
) -> pd.DataFrame:
    cols = {}
    for c, maps in full_maps.items():
        enc = apply_maps(X[c], maps, prior)
        for k in range(N_CLASSES):
            cols[f"te_{c}_{k}"] = enc[str(k)]
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)


def main() -> None:
    ds: Dataset = load_dataset()
    num_cols = [c for c in NUM_COLS if c in ds.train.columns]
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    params = load_params()
    print(f"Exact-value TE on {len(num_cols)} numerics (+{3 * len(num_cols)} cols)")

    def fit_fold(
        tr_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        te_tr, full_maps, prior = encode_fold(X, ds.y, tr_idx, num_cols, seed)
        X_tr = pd.concat([X.iloc[tr_idx], te_tr], axis=1)
        X_va = with_te(X.iloc[val_idx], full_maps, prior)
        X_te = with_te(X_test, full_maps, prior)
        model = LGBMClassifier(**{**params, "random_state": seed})
        model.fit(
            X_tr,
            ds.y[tr_idx],
            eval_set=[(X_va, ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof_proba, test_proba, fold_scores = bagged_cv(ds, fit_fold)
    finalize("lgbm_te", ds, oof_proba, test_proba, fold_scores)


if __name__ == "__main__":
    main()
