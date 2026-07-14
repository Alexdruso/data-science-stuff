"""Exact 3-way rule-triple TE (Day-11, encoding-crack probe).

The generator-forensics proof says the label channel is exactly 3-dimensional on
(stress x activity x EXACT sleep_duration). te13 encodes those columns UNIVARIATELY;
combo2 used BUCKETED sleep; rp used exact PAIRS (sleep-exact x stress, sleep-exact x
activity) — flat. The one encoding never built: the FULL cross-fitted 3-way posterior
P(y | stress x activity x EXACT sleep_duration) as a single column, at exact-sleep
resolution. It is the theoretically-optimal feature for the proven channel and a
genuine SUPERSET of the flat exact-pairs — not strictly bounded by them (a shallow
33-leaf HGBC need not reconstruct a 3-way interaction from univariate posteriors).

Adds ONE triple-key string column to the te13 TE view:
  sleep_duration-exact | stress_level | physical_activity_level
(~700 x 3 x 3 = ~6.3k cells, ~110 rows/cell; TargetEncoder smooth="auto" is
empirical-Bayes → shrinks sparse cells toward the marginal = the needed smoothing.)
NaN keeps the te13 'nan'-level convention. Same tuned HGBC, paired vs hgbc_te3_u_s42
(0.9504). Kill: paired < +0.0002.

Run: S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_tri.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

TRIPLE = ["sleep_duration", "stress_level", "physical_activity_level"]


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
    key = "tri_sleep_stress_activity"
    te_tr_view[key] = te_tr_view[TRIPLE].agg("|".join, axis=1)
    te_te_view[key] = te_te_view[TRIPLE].agg("|".join, axis=1)
    n_cells = te_tr_view[key].nunique()
    print(f"{key}: {n_cells} cells, ~{len(ds.train) / n_cells:.0f} rows/cell")
    te_cols = list(te_tr_view.columns)
    te_names = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASSES)]
    print(f"TE over {len(te_cols)} cols ({len(te_names)} features), tuned HGBC")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te3tri_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3tri", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3tri_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
