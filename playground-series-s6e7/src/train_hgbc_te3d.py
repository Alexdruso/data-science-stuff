"""ABLATION: v0.7 features (all-13 string TE) + OUR default HGBC config.

Isolates features-vs-config in the te3 gain. Original docstring follows.

 (repo notebook v0.7-hgbc-te.ipynb) — third TE attempt.

The two prior attempts missed his actual recipe. From the notebook source:
  1. TE is fit on ALL 13 RAW columns cast to STRINGS (numerics too — exact
     values; NaN becomes the 'nan'/'missing' string = its own TE level), via
     sklearn TargetEncoder(cv=5, smooth='auto', target_type='multiclass')
     -> 39 TE columns. Our attempts encoded the 7 numerics only.
  2. HGBC uses HIS TUNED config (lr 0.0627, max_iter 300, max_leaf_nodes 33,
     min_samples_leaf 298, l2 0.0289, max_bins 237, max_features 0.82,
     default-loss early stopping) — not our balanced-acc-scored default.
  3. Base features: raw numerics (NaN native) + cats as pandas Categorical
     with 'missing' filled level.

His anchor: OOF 0.95026 (his fold split), LB 0.95036 — no haircut.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc import HGBC_PARAMS
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

HGBC_CONFIG: dict[str, object] = {
    "learning_rate": 0.0627037115235577,
    "max_iter": 300,
    "max_leaf_nodes": 33,
    "min_samples_leaf": 298,
    "l2_regularization": 0.028912644384523085,
    "max_bins": 237,
    "max_features": 0.820265066682815,
    "early_stopping": True,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
}


def main() -> None:
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    # TE view: ALL columns as strings (numeric NaN -> 'nan' = its own level)
    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]
    print(f"v0.7 replica: TE over {len(raw_cols)} string-cast cols "
          f"({len(te_names)} TE features), tuned HGBC config")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [
                    base.reset_index(drop=True),
                    pd.DataFrame(blk, columns=te_names),
                ],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = HistGradientBoostingClassifier(**HGBC_PARAMS, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te3d_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3d", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3d_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
