"""PR-49: family-wise L2 for the pr42 stacker, evaluated on an honest outer row split.

    python src/stack_groupl2.py --ours $(cat results/ours_all80_*.txt)

Scaling group g's standardised columns by s_g is an L2 penalty of 1/s_g^2 on that group at the
global C. Outer 2-fold split (seed 0): on half A, one coordinate pass over the PR-48 groups with
s_g in {0.5, 1, 2}, each candidate scored by 5-fold LR on A; then LR fit on all of A with the
chosen and with uniform scales, both scored on half B; then swap. C scales with rows (sklearn
convention), so C is kept per-row equal to pr42's 0.01 at 630k training rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from source_ablation import group
from stack_noise import fit
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--C", type=float, default=0.01, help="pr42's C at 630k rows; rescaled per fit")
    p.add_argument("--grid", type=float, nargs="+", default=[0.5, 2.0])
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(np.float64)
    names, O, _ = load_pairs(a.ours, [a.public])
    O = np.clip(O, -30, 30)
    X = ((O - O.mean(0)) / O.std(0)).astype(np.float32)
    del O
    g = np.array([group(n) for n in names])
    groups = sorted(set(g))
    gi = {k: np.flatnonzero(g == k) for k in groups}
    n_ref = 0.9 * len(y)

    def lr_auc(tr: np.ndarray, va: np.ndarray, s: np.ndarray, warm: np.ndarray | None = None):
        Xs_tr = X[tr] * s
        w = fit(Xs_tr, y[tr], a.C * n_ref / len(tr), 0.0, warm)
        z = (X[va] * s) @ w[:-1] + w[-1]
        return z, w

    def inner(idx: np.ndarray, s: np.ndarray, warms: list) -> tuple[float, list]:
        oof, ws = np.zeros(len(idx)), []
        for j, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(idx, y[idx])):
            z, w = lr_auc(idx[tr], idx[va], s, warms[j] if warms else None)
            oof[va] = z
            ws.append(w)
        return roc_auc_score(y[idx], oof), ws

    halves = list(StratifiedKFold(2, shuffle=True, random_state=0).split(X, y))
    deltas = []
    for h, (A, B) in enumerate(halves):
        s = np.ones(X.shape[1], np.float32)
        best, warms = inner(A, s, [])
        print(f"half {h}: inner uniform {best:.6f}", flush=True)
        for k in groups:
            for v in a.grid:
                t = s.copy()
                t[gi[k]] = v
                auc, _ = inner(A, t, warms)
                if auc > best:
                    best, s = auc, t
            print(f"  {k:18s} -> s {s[gi[k]][0]:.1f}  inner {best:.6f}", flush=True)
        zu, _ = lr_auc(A, B, np.ones(X.shape[1], np.float32))
        zc, _ = lr_auc(A, B, s)
        du = roc_auc_score(y[B], zu)
        dc = roc_auc_score(y[B], zc)
        deltas.append(dc - du)
        print(f"half {h}: held-out uniform {du:.6f} chosen {dc:.6f} delta {dc - du:+.6f}", flush=True)
    print(f"mean held-out delta {np.mean(deltas):+.6f}", flush=True)


if __name__ == "__main__":
    main()
