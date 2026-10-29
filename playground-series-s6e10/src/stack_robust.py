"""PR-45: noise-robust stacker fit — downweight confidently-wrong training rows.

    python src/stack_robust.py --ours $(cat results/ours_all80_{nonnn,nn}.txt) --configs 0.02:0 0.05:0.3

Same single LR as PR-40/42 (standardised members, C 0.01, nested on --nest folds). Inside each
nesting fold the training part is split in two halves; an LR fit on one half scores the other, giving
every training row an honest p(observed label). Rows with p(observed) < t get weight w in the final
refit of that fold (config "t:w"; "1:1" = the unweighted control). Scoring uses every row, unweighted.
No validation-fold label touches the weights.
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


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, tol=1e-6, max_iter=20000)


def fold_fit(X, y, tr, va, T, C, configs, nest):
    p_obs = np.zeros(len(tr))
    for a, b in StratifiedKFold(2, shuffle=True, random_state=0).split(tr, y[tr]):
        p = 1 / (1 + np.exp(-_lr(C).fit(X[tr[a]], y[tr[a]]).decision_function(X[tr[b]])))
        p_obs[b] = np.where(y[tr[b]] == 1, p, 1 - p)
    out = {}
    for cfg in configs:
        t, w = (float(v) for v in cfg.split(":"))
        sw = np.where(p_obs < t, w, 1.0)
        m = _lr(C).fit(X[tr], y[tr], sample_weight=sw)
        out[cfg] = (m.decision_function(X[va]), m.decision_function(T) / nest, float((sw < 1).mean()))
    return va, out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--configs", nargs="+", default=["1:1", "0.02:0", "0.05:0.3"])
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--nest", type=int, default=5)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--write", default="", help="CONFIG=NAME")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, T = load_pairs(a.ours, [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    mu, sd = O.mean(0), O.std(0)
    X, XT = (O - mu) / sd, (T - mu) / sd
    print(f"{len(names)} members, C {a.C}, nest {a.nest}", flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=42).split(X, y))
    res = Parallel(n_jobs=a.jobs)(delayed(fold_fit)(X, y, tr, va, XT, a.C, a.configs, a.nest)
                                  for tr, va in folds)
    wcfg, wname = (a.write.split("=") + [""])[:2] if a.write else ("", "")
    for cfg in a.configs:
        oof, test = np.zeros(len(y)), np.zeros(len(T))
        frac = []
        for va, out in res:
            ov, tv, f = out[cfg]
            oof[va] = ov
            test += tv
            frac.append(f)
        print(f"{cfg:10s} pooled {roc_auc_score(y, oof):.6f}  downweighted {np.mean(frac):.4f}", flush=True)
        if cfg == wcfg:
            np.save(RESULTS_DIR / f"oof_{wname}.npy", oof)
            np.save(RESULTS_DIR / f"test_{wname}.npy", test)
            print(f"  saved {wname}")


if __name__ == "__main__":
    main()
