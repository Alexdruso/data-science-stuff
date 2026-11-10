"""Honest (nested) read of S8-1: pruning selected inside each nest split.

Per nest split (seeds 7/13/99 x 5 folds): fit LR C .01 on the train split over the v7 members,
keep the top-k by |coef| * sd, add the new members (unselected), refit, score the val split.
Compare with the free LR over v7 + new members (no pruning) and v7 alone.

    OMP_NUM_THREADS=4 python src/pool_pruned_nested.py ext/ds23 ext/... [--k 81]
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

from data_science_stuff.kaggle.stacking import load_oof_banks, rank_logit

R = Path(__file__).parent.parent / "results"
z = np.load(R / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
args = sys.argv[1:]
k = int(args[args.index("--k") + 1]) if "--k" in args else 81
dirs = [a for a in args if a.startswith("ext")]
mem = load_oof_banks(dirs, n_train=len(y), n_test=299984)
new = []
for name, (o, _) in sorted(mem.items()):
    if roc_auc_score(y, o) < 0.93:
        continue
    r = rank_logit(o)
    c = max(abs(np.corrcoef(r, Xo[:, j])[0, 1]) for j in range(Xo.shape[1]))
    if c < 0.9999 and all(abs(np.corrcoef(r, q)[0, 1]) < 0.9999 for _, q in new):
        new.append((name, r))
Xn = np.column_stack([r for _, r in new])
np.save(R / "pool_lab_new_1108.npy", Xn)
print(len(new), "new members", flush=True)
X_all = np.column_stack([Xo, Xn])
o_v7, o_all, o_pr = (np.zeros(len(y)) for _ in range(3))
seeds = (7, 13, 99)
for s in seeds:
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Xo, y):
        m = LogisticRegression(C=0.01, max_iter=5000).fit(Xo[trn], y[trn])
        o_v7[va] += m.decision_function(Xo[va]) / len(seeds)
        imp = np.abs(m.coef_[0]) * Xo[trn].std(0)
        keep = np.sort(np.argsort(-imp)[:k])
        Xp = np.column_stack([Xo[:, keep], Xn])
        mp = LogisticRegression(C=0.01, max_iter=5000).fit(Xp[trn], y[trn])
        o_pr[va] += mp.decision_function(Xp[va]) / len(seeds)
        ma = LogisticRegression(C=0.01, max_iter=5000).fit(X_all[trn], y[trn])
        o_all[va] += ma.decision_function(X_all[va]) / len(seeds)
    print(f"seed {s} done", flush=True)
a7, aa, ap = (roc_auc_score(y, o) for o in (o_v7, o_all, o_pr))
print(f"v7 {a7:.6f} | v7+new (free) {aa:.6f} ({aa - a7:+.6f}) | nested-pruned{k}+new {ap:.6f} ({ap - a7:+.6f})")
np.save(R / f"oof_lab_pruned{k}_new_1108.npy", o_pr)
