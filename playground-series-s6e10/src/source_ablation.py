"""PR-48: per-source ablation of the pr42 stack (std, C 0.01, nest 10) — drop each member group
in turn and re-fit (L-BFGS, warm-started from the full fit), then drop a union of groups.

    python src/source_ablation.py --ours $(cat results/ours_all80_*.txt) [--drop SRC ...] [--nest-seed 7]

Groups: public members by their `_pool` source prefix (text before "__"), own legs split into
NN (realmlp/tabm/nca/dae/mlp) and non-NN. Without --drop: control + every single-group drop.
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
from stack_noise import fit
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
NN = ("realmlp", "tabm", "nca", "dae", "mlp")


def group(name: str) -> str:
    if "/" in name:
        return name.split("/", 1)[1].split("__", 1)[0]
    return "own_nn" if any(k in name for k in NN) else "own_gbdt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--nest", type=int, default=10)
    p.add_argument("--nest-seed", type=int, default=42)
    p.add_argument("--drop", nargs="*", default=None, help="groups dropped together (union run)")
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(np.float64)
    names, O, T = load_pairs(a.ours, [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    mu, sd = O.mean(0), O.std(0)
    X, XT = ((O - mu) / sd).astype(np.float32), ((T - mu) / sd).astype(np.float32)
    del O, T
    g = np.array([group(n) for n in names])
    print(f"{len(names)} members; groups: " + ", ".join(f"{k} {int((g == k).sum())}" for k in sorted(set(g))),
          flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=a.nest_seed).split(X, y))

    def run(keep: np.ndarray, warm: list[np.ndarray] | None) -> tuple[float, list[np.ndarray], np.ndarray, np.ndarray]:
        cols = np.flatnonzero(keep)
        oof, test, ws = np.zeros(len(y)), np.zeros(len(XT)), []
        for j, (tr, va) in enumerate(folds):
            w0 = None if warm is None else np.r_[warm[j][:-1][cols], warm[j][-1]]
            w = fit(X[tr][:, cols], y[tr], a.C, 0.0, w0)
            full = np.zeros(X.shape[1] + 1)
            full[cols], full[-1] = w[:-1], w[-1]
            ws.append(full)
            oof[va] = X[va][:, cols] @ w[:-1] + w[-1]
            test += (XT[:, cols] @ w[:-1] + w[-1]) / a.nest
        return roc_auc_score(y, oof), ws, oof, test

    base, warm, oof, test = run(np.ones(len(names), bool), None)
    print(f"control {base:.6f}", flush=True)
    if a.drop is not None:
        auc, _, oof, test = run(~np.isin(g, a.drop), warm)
        print(f"drop {'+'.join(a.drop)}  {auc:.6f}  delta {auc - base:+.6f}", flush=True)
        if a.write:
            np.save(RESULTS_DIR / f"oof_{a.write}.npy", oof)
            np.save(RESULTS_DIR / f"test_{a.write}.npy", test)
        return
    for k in sorted(set(g)):
        auc, *_ = run(g != k, warm)
        print(f"drop {k:18s} ({int((g == k).sum()):3d})  {auc:.6f}  delta {auc - base:+.6f}", flush=True)


if __name__ == "__main__":
    main()
