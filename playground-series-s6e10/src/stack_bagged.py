"""PR-32: member-bagged LR stack. Each bag = LR (C) on a random subset of the members, nested on
the base folds exactly like stack_public.py; the deliverable is the mean of the bags' z-scored
logits (OOF and test z-scored with the bag's OOF mean/std). Dilutes any single member's weight.

    python src/stack_bagged.py --ours LEG ... --bags 32 --frac 0.5 [--write NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def one_bag(O, T, y, cols, C):
    oof, test = np.zeros(len(y)), np.zeros(len(T))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(O, y):
        sc = StandardScaler().fit(O[tr][:, cols])
        lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(O[tr][:, cols]), y[tr])
        oof[va] = lr.decision_function(sc.transform(O[va][:, cols]))
        test += lr.decision_function(sc.transform(T[:, cols])) / 5
    mu, sd = oof.mean(), oof.std()
    return (oof - mu) / sd, (test - mu) / sd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--exclude", nargs="*", default=[], help="substrings of member names to drop")
    p.add_argument("--bags", type=int, default=32)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, T = load_pairs(a.ours, [a.public])
    keep = [i for i, n in enumerate(names) if not any(e in n for e in a.exclude)]
    O, T = O[:, keep], T[:, keep]
    print(f"{len(keep)} members", flush=True)
    rng = np.random.default_rng(a.seed)
    subsets = [np.sort(rng.choice(len(keep), int(len(keep) * a.frac), replace=False)) for _ in range(a.bags)]
    res = Parallel(n_jobs=a.jobs)(delayed(one_bag)(O, T, y, c, a.C) for c in subsets)
    oof = np.mean([r[0] for r in res], axis=0)
    test = np.mean([r[1] for r in res], axis=0)
    for k in (1, 4, 8, 16, len(res)):
        if k <= len(res):
            print(f"bags {k}: {roc_auc_score(y, np.mean([r[0] for r in res[:k]], axis=0)):.6f}", flush=True)
    print(f"bagged stack OOF {roc_auc_score(y, oof):.6f}")
    if a.write:
        np.save(RESULTS_DIR / f"oof_{a.write}.npy", oof)
        np.save(RESULTS_DIR / f"test_{a.write}.npy", test)
        ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
        out = write_submission(SUBMISSIONS_DIR, f"{a.write}.csv", ids, TARGET, 1 / (1 + np.exp(-test)))
        print(f"→ {out}")


if __name__ == "__main__":
    main()
