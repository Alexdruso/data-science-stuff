"""PR-39: stacker penalty geometry — standardised members (ours) vs raw logits vs family-scaled raw
logits (the top public LB recipe: TabPFN columns ×3, other foundation models ×1.5, flat C).

    python src/stack_geometry.py --ours LEG ... --configs std:0.1 raw:0.003 fam:0.002 fam:0.003

Single LR per config, nested on the base folds (StratifiedKFold(5, True, 42)), so every row's
stack prediction is out of fold. Prints pooled and fold-mean AUC; --write saves one config.
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

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def family_scale(names: list[str]) -> np.ndarray:
    def fam(n: str) -> float:
        n = n.lower()
        if "tabpfn" in n or "pfn_" in n:
            return 3.0
        if "tfm-oof" in n or "tabicl" in n or "tabm" in n or "tabfm" in n or "limix" in n or "mitra" in n:
            return 1.5
        return 1.0
    return np.array([fam(n) for n in names])


def fold_fit(X, y, tr, va, T, C, nest=5):
    m = LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(X[tr], y[tr])
    return va, m.decision_function(X[va]), m.decision_function(T) / nest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--configs", nargs="+", default=["std:0.1", "raw:0.003", "fam:0.003"])
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--nest", type=int, default=5, help="stacker nesting folds (seed 42; 5 = the base folds)")
    p.add_argument("--nest-seed", type=int, default=42)
    p.add_argument("--exclude-public", nargs="*", default=[],
                   help="substrings dropped from public members only (own legs are kept)")
    p.add_argument("--write", default="", help="CONFIG=NAME, e.g. fam:0.003=pr39_fam")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, T = load_pairs(a.ours, [a.public])
    keep = [i for i, n in enumerate(names) if not ("/" in n and any(e in n for e in a.exclude_public))]
    names, O, T = [names[i] for i in keep], O[:, keep], T[:, keep]
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    sc = family_scale(names)
    print(f"{len(names)} members; family-scaled: {int((sc == 3).sum())} ×3, {int((sc == 1.5).sum())} ×1.5",
          flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=a.nest_seed).split(O, y))
    wcfg, wname = (a.write.split("=") + [""])[:2] if a.write else ("", "")
    for cfg in a.configs:
        kind, C = cfg.split(":")
        if kind == "std":
            mu, sd = O.mean(0), O.std(0)
            X, XT = (O - mu) / sd, (T - mu) / sd
        elif kind == "raw":
            X, XT = O, T
        else:
            X, XT = O * sc, T * sc
        res = Parallel(n_jobs=a.jobs)(delayed(fold_fit)(X, y, tr, va, XT, float(C), a.nest) for tr, va in folds)
        oof, test = np.zeros(len(y)), np.zeros(len(T))
        for va, ov, tv in res:
            oof[va] = ov
            test += tv
        fm = np.mean([roc_auc_score(y[va], oof[va]) for _, va in folds])
        print(f"{cfg:12s} pooled {roc_auc_score(y, oof):.6f}  fold-mean {fm:.6f}", flush=True)
        if cfg == wcfg:
            np.save(RESULTS_DIR / f"oof_{wname}.npy", oof)
            np.save(RESULTS_DIR / f"test_{wname}.npy", test)
            print(f"  saved {wname}")


if __name__ == "__main__":
    main()
