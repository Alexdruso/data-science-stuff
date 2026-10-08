"""Original-data teacher features (PR-19) — models fit on the ORIGINAL dataset only.

    python src/og_model.py            # fit + cache results/og_feats.npz

The competition target behaves like the original decision function plus label
noise: a model fit on the clean (near-separable, AUC 0.995) original labels
supplies detail that 700k noisy synthetic rows do not recover. No synthetic
label is used, so these are global features (leakage rule 2).

Columns (logits, in build_features() row order for train / test / original):
  og_xgb — XGBoost depth 8, 600 trees, lr 0.05 on the 21 raw columns
  og_lgb — LightGBM 63 leaves, 1500 trees, lr 0.03
For the original rows themselves (when they are appended as training rows) the
teacher's prediction is out-of-fold over the original (5-fold), otherwise it
would be a memorised near-copy of their label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import RATING_COLS
from matrices import Matrices, build_matrices

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
CACHE = RESULTS_DIR / "og_feats.npz"
RAW = ["age", "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes",
       *RATING_COLS, "gender", "customer_type", "type_of_travel", "class"]
OG_COLS = ["og_xgb", "og_lgb"]


def _models() -> dict[str, object]:
    return {
        "og_xgb": XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.5, tree_method="hist", device="cuda",
                                enable_categorical=True, random_state=0, verbosity=0),
        "og_lgb": LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=8,
                                 random_state=0),
    }


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).astype(np.float32)


def fit_cache() -> None:
    m = build_matrices(with_original=True)
    assert m.X_orig is not None and m.y_orig is not None
    xo, yo = m.X_orig[RAW], m.y_orig
    out: dict[str, np.ndarray] = {}
    for name in OG_COLS:
        model = _models()[name]
        model.fit(xo, yo)
        out[f"{name}_train"] = _logit(model.predict_proba(m.X[RAW])[:, 1])
        out[f"{name}_test"] = _logit(model.predict_proba(m.X_test[RAW])[:, 1])
        oof = np.zeros(len(xo))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(xo, yo):
            mm = _models()[name]
            mm.fit(xo.iloc[tr], yo[tr])
            oof[va] = mm.predict_proba(xo.iloc[va])[:, 1]
        out[f"{name}_orig"] = _logit(oof)
        from sklearn.metrics import roc_auc_score
        print(f"{name}: AUC on synthetic train {roc_auc_score(m.y, out[f'{name}_train']):.5f}  "
              f"original OOF {roc_auc_score(yo, oof):.5f}", flush=True)
    np.savez(CACHE, **out)
    print(f"cached → {CACHE}")


def add_og(m: Matrices, x_orig: pd.DataFrame | None = None, cols: list[str] | None = None) -> None:
    """Attach cached teacher logits to m.X / m.X_test (and x_orig in place, if given)."""
    z = np.load(CACHE)
    for c in cols or OG_COLS:
        m.X[c] = z[f"{c}_train"]
        n = len(z[f"{c}_test"])
        if len(m.X_test) == n:
            m.X_test[c] = z[f"{c}_test"]
        else:  # X_test still carries the original rows appended (route-global pass)
            m.X_test[c] = np.concatenate([z[f"{c}_test"], z[f"{c}_orig"]])
        if x_orig is not None:
            x_orig[c] = z[f"{c}_orig"]


if __name__ == "__main__":
    fit_cache()
