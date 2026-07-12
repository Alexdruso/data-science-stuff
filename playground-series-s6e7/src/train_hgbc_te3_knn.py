"""hgbc_te3 + kNN label-distribution features (Day-10 PM, strategist #1).

Consumes the cached fold-safe kNN blocks (knn_features.py — class distribution
among k in {10,30,100} nearest neighbours in the 6 leakage numerics) as 9
extra numeric columns on top of the standard te13 design. Paired vs
hgbc_te3_u_s42 (0.9504). Field-verified channel: honest solo 0.95054
(r1ck.sync). Kill: paired < +0.0002 AND LR-seat negative.

Run (unrepaired namespace, after knn_features.py has cached the folds):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_knn.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from knn_features import KNN_NAMES, knn_block_for_fold
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    print(f"v0.7 design + {len(KNN_NAMES)} kNN label-distribution cols")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        knn_tr, knn_te, _ = knn_block_for_fold(ds, tr_idx, seed, fold)
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base, blk, knn):  # noqa: ANN001, ANN202
            return pd.concat(
                [
                    base.reset_index(drop=True),
                    pd.DataFrame(blk, columns=te_names),
                    pd.DataFrame(knn, columns=KNN_NAMES),
                ],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr, knn_tr[tr_idx])
        X_va = assemble(X.iloc[val_idx], te_va, knn_tr[val_idx])
        X_te = assemble(X_test, te_te, knn_te)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te3knn_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3knn", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3knn_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
