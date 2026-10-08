"""Auxiliary-task features (PR-20) — predict each column from the other 20, no labels.

    python src/aux_feats.py          # cache results/aux_feats.npz

Idea from the public notebook goodpjw2008 "Auxiliary-Task Features" (+0.0001 on a
single tree model there). For each of the 13 ratings and the 3 binary/ternary
categoricals, an XGBoost predicts the column from the other 20 raw columns over
train + test (+ original rows, so they can be appended as training rows),
5-fold so no row is predicted by a model that saw it. Kept per row:
  aux_ev_<rating>  expected rating  sum_k k·P(k)
  aux_p_<col>      probability of the value the row actually has
  aux_sum_logp     sum of log aux_p (row pseudo-log-likelihood)
No satisfaction label is involved, so these are global features (leakage rule 2).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import RATING_COLS
from matrices import Matrices, build_matrices
from og_model import RAW

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
CACHE = RESULTS_DIR / "aux_feats.npz"
TARGETS = [*RATING_COLS, "class", "type_of_travel", "customer_type"]


def fit_cache() -> None:
    m = build_matrices(with_original=True)
    assert m.X_orig is not None
    full = pd.concat([m.X[RAW], m.X_test[RAW], m.X_orig[RAW]], ignore_index=True)
    coded = full.copy()
    for c in ("gender", "customer_type", "type_of_travel", "class"):
        coded[c] = coded[c].cat.codes
    out = pd.DataFrame(index=full.index)
    t0 = time.time()
    for c in TARGETS:
        vals = np.sort(coded[c].unique())
        yc = np.searchsorted(vals, coded[c].to_numpy())
        k = len(vals)
        others = [f for f in RAW if f != c]
        P = np.zeros((len(full), k), dtype=np.float32)
        for a, b in KFold(5, shuffle=True, random_state=0).split(coded):
            model = XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.7, tree_method="hist", device="cuda", random_state=0,
                objective="multi:softprob" if k > 2 else "binary:logistic", verbosity=0,
            )
            model.fit(coded.iloc[a][others], yc[a])
            pb = model.predict_proba(coded.iloc[b][others])
            P[b] = pb
            del model
        out[f"aux_p_{c}"] = P[np.arange(len(full)), yc]
        if c in RATING_COLS:
            out[f"aux_ev_{c}"] = P @ vals.astype(np.float32)
        print(f"{c:36s} k={k} acc {(P.argmax(1) == yc).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    p_cols = [c for c in out if c.startswith("aux_p_")]
    out["aux_sum_logp"] = np.log(np.clip(out[p_cols].to_numpy(), 1e-6, 1)).sum(1)
    n_tr, n_te = len(m.X), len(m.X_test)
    arr = out.to_numpy(np.float32)
    np.savez(CACHE, cols=np.array(out.columns), train=arr[:n_tr], test=arr[n_tr:n_tr + n_te],
             orig=arr[n_tr + n_te:])
    print(f"cached {arr.shape} → {CACHE}")


def add_aux(m: Matrices, x_orig: pd.DataFrame | None = None, which: str = "ev") -> None:
    """Attach cached aux features (`ev` = the 13 expected values; `all` = everything)."""
    z = np.load(CACHE, allow_pickle=True)
    cols = list(z["cols"])
    keep = [i for i, c in enumerate(cols) if which == "all" or c.startswith("aux_ev_")]
    test = z["test"] if len(m.X_test) == len(z["test"]) else np.vstack([z["test"], z["orig"]])
    for i in keep:
        m.X[cols[i]] = z["train"][:, i]
        m.X_test[cols[i]] = test[:, i]
        if x_orig is not None:
            x_orig[cols[i]] = z["orig"][:, i]


if __name__ == "__main__":
    fit_cache()
