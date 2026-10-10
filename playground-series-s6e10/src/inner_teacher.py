"""Honest strong tree teacher for fold-1 distillation tests: inner 5-fold LGBM on the 69
exported columns of fold 1's TRAINING rows only (route TE already inner-OOF there). Saves a
full-length array (0.5 on fold-1 val rows, never read) usable as --soft-teacher.

    python src/inner_teacher.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RESULTS = Path(__file__).resolve().parents[1] / "results"
P = dict(objective="binary", learning_rate=0.02, num_leaves=127, min_data_in_leaf=200,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
         num_threads=12, seed=0)


def main() -> None:
    z = np.load(RESULTS / "tfm_fold1.npz", allow_pickle=True)
    x, y = z["x_tr"], z["y_tr"]
    n = len(y) + len(z["y_va"])
    tr_idx = np.setdiff1d(np.arange(n), z["va_idx"])
    p = np.zeros(len(y))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(x, y):
        b = lgb.train(P, lgb.Dataset(x[ia], y[ia]), 1500)
        p[ib] = b.predict(x[ib])
        print(f"  inner fold AUC {roc_auc_score(y[ib], p[ib]):.5f}", flush=True)
    print(f"inner teacher AUC {roc_auc_score(y, p):.5f}", flush=True)
    full = np.full(n, 0.5)
    full[tr_idx] = p
    np.save(RESULTS / "oof_inner_lgbm_f1.npy", full)


if __name__ == "__main__":
    main()
