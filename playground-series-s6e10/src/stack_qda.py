"""PR-52: generative combiner — shrunk QDA on principal components of the member logits.

    python src/stack_qda.py --ours $(cat results/ours_all80_nonnn.txt results/ours_all80_nn.txt) \
        --configs 10:0.1 30:0.1 60:0.3

Same members and nesting as pr42 (364 standardised clipped logits, StratifiedKFold(10, True, 42)).
Per nesting fold: PCA(k) + QDA(reg_param r) fit on the training rows, class log-ratio on the
held-out rows. The LR stack is the discriminative twin of LDA; QDA adds the class-covariance
(quadratic) terms. Read = 5-fold 2-feature LR (pr42 logit, QDA score) vs pr42 alone (screen_raw).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def fold_fit(X, y, tr, va, XT, k, r, nest):
    pca = PCA(k, random_state=0).fit(X[tr])
    q = QuadraticDiscriminantAnalysis(reg_param=r).fit(pca.transform(X[tr]), y[tr])
    return va, q.decision_function(pca.transform(X[va])), q.decision_function(pca.transform(XT)) / nest


def two_feature(y, s, c):
    X = np.column_stack([s, c])
    o = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        o[b] = LogisticRegression(max_iter=500).fit(X[a], y[a]).decision_function(X[b])
    return roc_auc_score(y, o)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--stack", default="pr42_lr_allown_c01_n10")
    p.add_argument("--configs", nargs="+", default=["10:0.1", "30:0.1", "60:0.3"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--nest", type=int, default=10)
    p.add_argument("--write", default="", help="CONFIG=NAME")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    _, O, T = load_pairs(a.ours, [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    mu, sd = O.mean(0), O.std(0)
    X, XT = (O - mu) / sd, (T - mu) / sd
    s = np.load(RESULTS_DIR / f"oof_{a.stack}.npy")
    base = two_feature(y, s, np.random.default_rng(0).normal(size=len(y)))
    print(f"{X.shape[1]} members; pr42 + noise {base:.6f}", flush=True)
    folds = list(StratifiedKFold(a.nest, shuffle=True, random_state=42).split(X, y))
    wcfg, wname = (a.write.split("=") + [""])[:2] if a.write else ("", "")
    for cfg in a.configs:
        k, r = cfg.split(":")
        res = Parallel(n_jobs=a.jobs)(
            delayed(fold_fit)(X, y, tr, va, XT, int(k), float(r), a.nest) for tr, va in folds)
        oof, test = np.zeros(len(y)), np.zeros(len(T))
        for va, ov, tv in res:
            oof[va] = ov
            test += tv
        d = two_feature(y, s, oof) - base
        print(f"{cfg:8s} solo {roc_auc_score(y, oof):.6f} corr {np.corrcoef(s, oof)[0, 1]:.4f} "
              f"delta {d:+.6f}", flush=True)
        if cfg == wcfg:
            np.save(RESULTS_DIR / f"oof_{wname}.npy", oof)
            np.save(RESULTS_DIR / f"test_{wname}.npy", test)


if __name__ == "__main__":
    main()
