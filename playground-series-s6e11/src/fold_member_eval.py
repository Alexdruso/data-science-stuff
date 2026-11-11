"""Fold-level read of new members inside the full cached member matrix (not a single pooled logit).

    python src/fold_member_eval.py _clm_any_ep10_first _clm_any_ep10_last [--fold 1] [--joint]

Rows of one member-CV fold only (the rows a fold-pilot has scores for). Base = cached v7 matrix +
megayak LM (= pool_v11 members); nested LR (C scaled by 5 for the 5x fewer rows), 3 nest seeds, with
and without the new column(s). Unlike `lm_pool_eval.py`, the stack can re-weight the members the new
column correlates with (megayak's LM reads -0.00013 on a single pooled logit but +0.00004 in the pool).
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.stacking import nested_lr_stack, rank_logit

R = Path(__file__).parent.parent / "results"
ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*")
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--joint", action="store_true", help="also add all dirs together")
ap.add_argument("--c", type=float, default=0.05)
ap.add_argument("--no_mega", action="store_true")
a = ap.parse_args()
z = np.load(R / "pool_lab.npz")
y = z["y"]
m = fold_assignment(y) == a.fold
ys = y[m]
X = np.column_stack([rank_logit(c) for c in z["Xo"][m].T])
mega = rank_logit(np.load(R.parent / "ext/ds23/megayak-lm/lm_oof.npy")[m])
b7 = nested_lr_stack(X, ys, c=a.c).auc
B = X if a.no_mega else np.column_stack([X, mega])
b = nested_lr_stack(B, ys, c=a.c).auc
print(f"fold {a.fold}: v7 members {b7:.6f}  +megayak {b - b7:+.6f}  (base {b:.6f})", flush=True)
cols = []
for d in a.dirs:
    v = rank_logit(np.load(R / d / f"f{a.fold}_val.npy"))
    cols.append(v)
    print(f"  {d:40s} solo {roc_auc_score(ys, v):.5f}  delta {nested_lr_stack(np.column_stack([B, v]), ys, c=a.c).auc - b:+.6f}"
          f"  rho(mega) {np.corrcoef(v, mega)[0, 1]:.4f}", flush=True)
if a.joint and len(cols) > 1:
    print(f"  joint ({len(cols)}) delta {nested_lr_stack(np.column_stack([B, *cols]), ys, c=a.c).auc - b:+.6f}", flush=True)
