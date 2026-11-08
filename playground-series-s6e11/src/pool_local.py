"""S8-2: local re-stack of the contested region on the cached v7 matrix.

AUC is decided by how positives rank against the negatives near them; ~80% of rows are easy
negatives that dominate the global log-loss fit. Per nest split: global LR (C .01) on all train
rows; region = train rows with global score >= quantile q; local LR fit on region rows only;
val rows in the region get the local score mapped onto the global scale (linear fit of global
on local over the train region), others keep the global score. Same nest seeds as lr_stack.

    OMP_NUM_THREADS=4 python src/pool_local.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

z = np.load(Path(__file__).parent.parent / "results" / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
QS = (0.8, 0.9, 0.95)
CL = (0.01, 0.1)
base = np.zeros(len(y))
out = {(q, c): np.zeros(len(y)) for q in QS for c in CL}
seeds = (7, 13, 99)
for s in seeds:
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Xo, y):
        g = LogisticRegression(C=0.01, max_iter=5000).fit(Xo[trn], y[trn])
        gt, gv = g.decision_function(Xo[trn]), g.decision_function(Xo[va])
        base[va] += gv / len(seeds)
        for q in QS:
            thr = np.quantile(gt, q)
            rt, rv = gt >= thr, gv >= thr
            for c in CL:
                loc = LogisticRegression(C=c, max_iter=5000).fit(Xo[trn][rt], y[trn][rt])
                lt = loc.decision_function(Xo[trn][rt])
                a, b = np.polyfit(lt, gt[rt], 1)
                f = gv.copy()
                f[rv] = a * loc.decision_function(Xo[va][rv]) + b
                out[(q, c)][va] += f / len(seeds)
        print(f"seed {s} split done", flush=True)
b = roc_auc_score(y, base)
print(f"global {b:.6f}")
for (q, c), v in out.items():
    print(f"  region q {q} C {c}: {roc_auc_score(y, v):.6f}  delta {roc_auc_score(y, v) - b:+.6f}")
