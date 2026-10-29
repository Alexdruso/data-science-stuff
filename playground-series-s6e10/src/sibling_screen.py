"""Exploratory: do within-route near-siblings (synthetic rows sharing >= m of 21 exact column
values with a row, same flight_distance = route ID) carry label information beyond the stack?

For each train row: shrunk mean of the siblings' stack residuals (y - p_stack), m = 17..21.
Scored as stack logit + beta * feature, beta fit on one half of the rows, AUC on the other
(both directions). Siblings' residuals come from the honest nested stack OOF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
MS = (21, 20, 19, 18, 17)


def block(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(n, len(MS), 2): residual sum and count of siblings with >= m matches."""
    out = np.zeros((len(x), len(MS), 2))
    for i in range(0, len(x), 512):
        eq = (x[i:i + 512, None, :] == x[None, :, :]).sum(2)
        for j in range(eq.shape[0]):
            eq[j, i + j] = -1
        for t, m in enumerate(MS):
            s = eq >= m
            out[i:i + 512, t, 0] = s @ r
            out[i:i + 512, t, 1] = s.sum(1)
    return out


def main() -> None:
    stack = sys.argv[1] if len(sys.argv) > 1 else "pr42_lr_allown_c01_n10_avg2"
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = tr[TARGET].to_numpy().astype(float)
    p = np.load(RESULTS_DIR / f"oof_{stack}.npy")
    if p.min() < 0 or p.max() > 1:
        p = 1 / (1 + np.exp(-p))
    s = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    cols = [c for c in tr.columns[1:23] if c not in ("flight_distance", TARGET)]
    x = np.column_stack([tr[c].fill_null(-1).cast(pl.Int64).to_numpy() if tr[c].dtype != pl.Utf8
                         else tr[c].cast(pl.Categorical).to_physical().to_numpy() for c in cols])
    x = x.astype(np.int32)
    fd = tr["flight_distance"].to_numpy()
    r = y - p
    groups = [np.where(fd == v)[0] for v in np.unique(fd)]
    groups.sort(key=len, reverse=True)
    res = Parallel(n_jobs=8)(delayed(block)(x[g], r[g]) for g in groups)
    agg = np.zeros((len(y), len(MS), 2))
    for g, o in zip(groups, res):
        agg[g] = o
    rng = np.random.default_rng(0)
    half = rng.random(len(y)) < 0.5
    print(f"{len(cols)} cols; stack AUC {roc_auc_score(y, s):.6f}")
    for t, m in enumerate(MS):
        cnt = agg[:, t, 1]
        print(f"m>={m}: rows with >=1 sibling {np.mean(cnt > 0):.3f}, mean count {cnt.mean():.2f}")
        for k in (2, 10, 50):
            f = agg[:, t, 0] / (cnt + k)
            d = []
            for h in (half, ~half):
                X = np.column_stack([s, f])
                lr = LogisticRegression(C=1e3).fit(X[h], y[h])
                d.append(roc_auc_score(y[~h], lr.decision_function(X[~h])) - roc_auc_score(y[~h], s[~h]))
            print(f"   k={k}: delta {np.mean(d):+.6f} ({d[0]:+.6f}, {d[1]:+.6f})", flush=True)


if __name__ == "__main__":
    main()
