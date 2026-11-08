"""S8-4: shallow LightGBM boosted from the nested LR stack score over the v7 members.

init_score = nested LR OOF decision (lr_stack, seeds 7/13/99; honest per row); a 4-leaf GBDT
on the 171 rank-logit members learns member x member corrections. Outer 5-fold CV (seed 2024),
early stopping on an inner 10% of each outer train split. Delta vs the LR OOF itself.

    OMP_NUM_THREADS=4 python src/pool_gbdt_resid.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from pool_lab import lr_stack

z = np.load(Path(__file__).parent.parent / "results" / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
auc0, s0 = lr_stack(Xo, y)
print(f"LR nested {auc0:.6f}", flush=True)
for leaves, lr, mcs in ((4, 0.02, 2000), (8, 0.02, 4000)):
    out = s0.copy()
    its = []
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=2024).split(Xo, y):
        a, b = train_test_split(trn, test_size=0.1, stratify=y[trn], random_state=0)
        dtr = lgb.Dataset(Xo[a], y[a], init_score=s0[a])
        dva = lgb.Dataset(Xo[b], y[b], init_score=s0[b], reference=dtr)
        m = lgb.train(dict(objective="binary", metric="auc", num_leaves=leaves, learning_rate=lr,
                           min_child_samples=mcs, feature_fraction=0.5, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l2=10.0, num_threads=4, verbose=-1, seed=0),
                      dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        out[va] = s0[va] + m.predict(Xo[va], raw_score=True, num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    a1 = roc_auc_score(y, out)
    print(f"  leaves {leaves} mcs {mcs}: {a1:.6f}  delta {a1 - auc0:+.6f}  best_iter {its}", flush=True)
