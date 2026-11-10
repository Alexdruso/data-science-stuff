"""S8-3: does the pool's LM weight suit the less-noisy test-side LM inputs?

On test each own LM is an average of 5 fold models (x K orders); on OOF it is one model. Fold 1
has three independent ep10 runs (q0 = stored, q1, q2 from PR-15), so: fit the free LR (C .01) on
folds 2-5 of the cached v7 matrix, then score fold 1 with the ep10 column = (a) q0 as stored,
(b) avg(q0,q1,q2) quantile-mapped onto q0's fold-1 values (ranks change, marginal does not),
(c) (b) with the ep10 coefficient scaled by lambda. b > a: the test side is better than OOF
says; c(lambda>1) > b: an attenuation correction would pay.

    OMP_NUM_THREADS=4 python src/attenuation_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment

R = Path(__file__).parent.parent / "results"
z = np.load(R / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
names = json.load(open(R / "pool_lab_v7_names.json"))
c = names.index("lm_distilgpt2_ep10")
tr = build_features(pl.read_csv(R.parent / "data" / "train.csv"))
folds = fold_assignment(tr[TARGET].to_numpy())
assert (tr[TARGET].to_numpy() == y).all()
va = np.where(folds == 1)[0]
q = [np.load(R / d / "f1_val.npy") for d in ("_lm_distilgpt2_ep10", "_lm_distilgpt2_ep10_q1",
                                             "_lm_distilgpt2_ep10_q2")]
print("q corr", np.corrcoef(q)[0, 1:], "stored-col vs q0 spearman",
      np.corrcoef(np.argsort(np.argsort(Xo[va, c])), np.argsort(np.argsort(q[0])))[0, 1])
avg = np.mean(q, 0)
srt = np.sort(Xo[va, c])
mapped = srt[np.argsort(np.argsort(avg))]
trn = folds != 1
m = LogisticRegression(C=0.01, max_iter=5000).fit(Xo[trn], y[trn])
w = m.coef_[0][c]
Xa = Xo[va].copy()
sa = m.decision_function(Xa)
Xb = Xa.copy(); Xb[:, c] = mapped
sb = m.decision_function(Xb)
ya = y[va]
print(f"ep10 coef {w:.3f}; fold-1 AUC (a) q0 {roc_auc_score(ya, sa):.6f}  (b) avg3 {roc_auc_score(ya, sb):.6f} "
      f"({roc_auc_score(ya, sb) - roc_auc_score(ya, sa):+.6f})")
for lam in (0.8, 1.2, 1.5, 2.0, 3.0):
    sc = sb + (lam - 1) * w * Xb[:, c]
    sq = sa + (lam - 1) * w * Xa[:, c]
    print(f"  lambda {lam}: (c) avg3 {roc_auc_score(ya, sc):.6f} ({roc_auc_score(ya, sc) - roc_auc_score(ya, sb):+.6f} vs b)"
          f"   q0 {roc_auc_score(ya, sq) - roc_auc_score(ya, sa):+.6f} vs a")
