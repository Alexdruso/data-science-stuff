"""PR-30: label-noise cleaning, fold 1 (screen). Inner 3-fold LGBM on the fold's training rows
gives each training row an honest p; arms: plain / down-weight rows with |y - p| > 0.8 /
soft labels 0.7·y + 0.3·p. Same fixed-round LGBM everywhere; val rows are never touched.

    python src/label_clean.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RESULTS = Path(__file__).resolve().parents[1] / "results"
P = dict(objective="cross_entropy", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
         num_threads=12, seed=0)


def fit(x, t, w, xs, rounds):
    b = lgb.train(P, lgb.Dataset(x, t, weight=w), rounds)
    return [b.predict(q, raw_score=True) for q in xs]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    P["num_threads"] = a.threads
    z = np.load(RESULTS / "tfm_fold1.npz", allow_pickle=True)
    x, y, xv, yv = z["x_tr"], z["y_tr"].astype(float), z["x_va"], z["y_va"]
    p = np.zeros(len(y))
    for ia, ib in StratifiedKFold(a.inner, shuffle=True, random_state=0).split(x, y):
        p[ib] = 1 / (1 + np.exp(-fit(x[ia], y[ia], None, [x[ib]], 800)[0]))
    print(f"inner OOF AUC {roc_auc_score(y, p):.5f}; |y-p|>0.8: {(np.abs(y - p) > 0.8).mean():.4f}", flush=True)
    if a.alphas:
        for al in a.alphas:
            pv = fit(x, (1 - al) * y + al * p, None, [xv], 1500)[0]
            print(f"soft a={al} inner={a.inner}: val AUC {roc_auc_score(yv, pv):.5f}", flush=True)
            np.save(RESULTS / f"probe_lc_soft{al}_i{a.inner}_f1.npy", np.stack([z["va_idx"].astype(float), pv]))
        return
    arms = {"plain": (y, None),
            "downweight": (y, np.where(np.abs(y - p) > 0.8, 0.3, 1.0)),
            "soft": (0.7 * y + 0.3 * p, None)}
    for name, (t, w) in arms.items():
        pv = fit(x, t, w, [xv], 1500)[0]
        print(f"{name}: val AUC {roc_auc_score(yv, pv):.5f}", flush=True)
        np.save(RESULTS / f"probe_lc_{name}_f1.npy", np.stack([z["va_idx"].astype(float), pv]))


if __name__ == "__main__":
    main()
