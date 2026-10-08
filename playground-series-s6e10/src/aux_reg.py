"""PR-23: label-free regression aux features — E[flight_distance | other 20], E[age | other 20].

    python src/aux_reg.py        # cache results/aux_reg.npz

Like `aux_feats.py` (5-fold over train + test + original, no satisfaction label),
for the two numeric keys: the expected value and the row's residual. A passenger
atypical for their route (fd − E[fd]) is information the route key alone lacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from matrices import Matrices, build_matrices
from og_model import RAW

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
CACHE = RESULTS_DIR / "aux_reg.npz"
TARGETS = ["flight_distance", "age"]


def fit_cache() -> None:
    m = build_matrices(with_original=True)
    assert m.X_orig is not None
    full = pd.concat([m.X[RAW], m.X_test[RAW], m.X_orig[RAW]], ignore_index=True)
    out = {}
    for c in TARGETS:
        others = [f for f in RAW if f != c]
        y = full[c].to_numpy(float)
        pred = np.zeros(len(full))
        for a, b in KFold(5, shuffle=True, random_state=0).split(full):
            mdl = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=127, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=12)
            mdl.fit(full.iloc[a][others], y[a])
            pred[b] = mdl.predict(full.iloc[b][others])
        out[f"auxr_ev_{c}"] = pred
        out[f"auxr_res_{c}"] = y - pred
        print(f"{c}: R2 {1 - np.var(y - pred) / np.var(y):.3f}", flush=True)
    df = pd.DataFrame(out).astype(np.float32)
    n_tr, n_te = len(m.X), len(m.X_test)
    arr = df.to_numpy()
    np.savez(CACHE, cols=np.array(df.columns), train=arr[:n_tr], test=arr[n_tr:n_tr + n_te], orig=arr[n_tr + n_te:])
    print(f"cached → {CACHE}")


def add_aux_reg(m: Matrices, x_orig: pd.DataFrame | None = None) -> None:
    z = np.load(CACHE, allow_pickle=True)
    cols = list(z["cols"])
    test = z["test"] if len(m.X_test) == len(z["test"]) else np.vstack([z["test"], z["orig"]])
    for i, c in enumerate(cols):
        m.X[c] = z["train"][:, i]
        m.X_test[c] = test[:, i]
        if x_orig is not None:
            x_orig[c] = z["orig"][:, i]


if __name__ == "__main__":
    fit_cache()
