"""Rule-pair TE at EXACT-VALUE resolution (Day-10 PM, strategist #2).

The generator learned a SIGMOID at the sleep 6/7 cut points (field notebook,
2026-07-12): rows at sleep 6.01-6.10 x stress=high are 20-40% unhealthy — a
joint statistic univariate te13 cannot represent and 33-leaf trees can't
compose. Both prior kills straddle this without covering it: combo2 used
sleep BUCKETS (coarsens away the band); te3p used exact values but only on
generator-COUPLING pairs (step x activity etc.), never the RULE pairs.

Adds two pair-key string columns to the te13 TE view:
  sleep_duration-exact x stress_level
  sleep_duration-exact x physical_activity_level
(NaN keeps the te13 'nan'-level convention on both sides.) Same tuned HGBC,
paired vs hgbc_te3_u_s42 (0.9504). Kill: paired < +0.0002.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_rp.py
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

PAIRS = [
    ("sleep_duration", "stress_level"),
    ("sleep_duration", "physical_activity_level"),
]


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
    for a, b in PAIRS:
        te_tr_view[f"rp_{a}__{b}"] = te_tr_view[a] + "|" + te_tr_view[b]
        te_te_view[f"rp_{a}__{b}"] = te_te_view[a] + "|" + te_te_view[b]
        n_cells = te_tr_view[f"rp_{a}__{b}"].nunique()
        print(f"rp_{a}__{b}: {n_cells} cells, "
              f"~{len(ds.train) / n_cells:.0f} rows/cell")
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
        ds, fit_fold, ckpt_name=f"hgbc_te3rp_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3rp", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3rp_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
