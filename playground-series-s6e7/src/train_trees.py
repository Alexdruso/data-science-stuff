"""ExtraTrees + RandomForest — bagging-tree diversity probes (zoo Z5, PS S6E7).

Bagging + randomized splits is a genuinely different prior from boosting
(variance- vs bias-reduction) and has never been priced on this dataset. sklearn
forests reject NaN: numerics are median-imputed with TRAIN-FOLD stats + explicit
missingness indicators; the 6 categoricals are one-hot with a NaN dummy. The
label is a coarse 3-feature rule, so min_samples_leaf=50 loses nothing and keeps
600 trees on 552k rows inside RAM.

Run (repaired surface, zoo protocol):
  S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_trees.py
Gate: diag_mlp_transfer signature vs the repaired breadth core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, Dataset, finalize, load_dataset
from zoo_common import clear_ckpt, zoo_cv

MODELS = {
    "extratrees": lambda seed: ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    ),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    ),
}


def encode(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-independent parts: raw numerics (NaN kept), OHE cats, missingness block."""
    num_cols = [c for c in ds.feature_cols if c not in ds.cat_cols]
    combined = pd.concat(
        [ds.train[ds.cat_cols], ds.test[ds.cat_cols]], ignore_index=True
    )
    ohe = pd.get_dummies(combined, columns=ds.cat_cols, dummy_na=True, dtype=np.float32)
    n = len(ds.train)
    num_tr = ds.train[num_cols].to_numpy(dtype=np.float64)
    num_te = ds.test[num_cols].to_numpy(dtype=np.float64)
    miss_tr = np.isnan(num_tr).astype(np.float32)
    miss_te = np.isnan(num_te).astype(np.float32)
    Xc_tr = np.hstack([miss_tr, ohe.iloc[:n].to_numpy(np.float32)])
    Xc_te = np.hstack([miss_te, ohe.iloc[n:].to_numpy(np.float32)])
    return num_tr, num_te, Xc_tr, Xc_te


def main() -> None:
    ds = load_dataset()
    num_tr, num_te, Xc_tr, Xc_te = encode(ds)
    print(f"dense: {num_tr.shape[1]} numerics + {Xc_tr.shape[1]} indicator/ohe cols")

    for name, make in MODELS.items():

        def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
            med = np.nanmedian(num_tr[tr_idx], axis=0)
            Xtr = np.hstack(
                [np.where(np.isnan(num_tr), med, num_tr).astype(np.float32), Xc_tr]
            )
            Xte = np.hstack(
                [np.where(np.isnan(num_te), med, num_te).astype(np.float32), Xc_te]
            )
            m = make(seed)  # noqa: B023 - loop var bound per iteration below
            m.fit(Xtr[tr_idx], ds.y[tr_idx])
            assert list(m.classes_) == list(range(N_CLASSES)), m.classes_
            val_proba = m.predict_proba(Xtr[val_idx])
            test_proba = m.predict_proba(Xte)
            del m
            return val_proba, test_proba

        print(f"\n=== {name} ===")
        oof, test, fold_scores = zoo_cv(
            ds, fit_fold, ckpt_name=f"{name}_s{SEEDS[0]}", seed=SEEDS[0]
        )
        finalize(name, ds, oof, test, fold_scores)
        clear_ckpt(f"{name}_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
