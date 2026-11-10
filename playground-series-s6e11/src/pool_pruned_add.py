"""S8-1: marginal value of new banks on the importance-pruned v7 pool (81 members).

Importance = |coef| * sd(column) from a full C .01 fit on the v7 matrix (as S7-4); keep the top
81. New members (deduped vs the full v7 matrix) are then added unselected, so their delta on the
pruned pool is a fair read of their marginal value there.

    python src/pool_pruned_add.py ext/ds23 ext/ds25_conv/subodh ...
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from data_science_stuff.kaggle.stacking import load_oof_banks, nested_lr_stack, rank_logit

z = np.load(Path(__file__).parent.parent / "results" / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
keep_n = 81
m = LogisticRegression(C=0.01, max_iter=5000).fit(Xo, y)
imp = np.abs(m.coef_[0]) * Xo.std(0)
keep = np.sort(np.argsort(-imp)[:keep_n])
Xp = Xo[:, keep]
mem = load_oof_banks(dirs, n_train=len(y), n_test=299984)
new = []
for k, (o, _) in sorted(mem.items()):
    if roc_auc_score(y, o) < 0.93:
        continue
    r = rank_logit(o)
    c = max(abs(np.corrcoef(r, Xo[:, j])[0, 1]) for j in range(Xo.shape[1]))
    if c < 0.9999 and all(abs(np.corrcoef(r, q)[0, 1]) < 0.9999 for _, q in new):
        new.append((k, r))
print(len(new), "new members", flush=True)
b_full = 0.9526657545740411
b_p = nested_lr_stack(Xp, y).auc
print(f"pruned81 {b_p:.6f} ({b_p - b_full:+.6f} vs v7)", flush=True)
groups: dict[str, list[np.ndarray]] = {}
for k, r in new:
    groups.setdefault(k.split(":")[0] if "megayak" not in k else "megayak", []).append(r)
for g, rs in groups.items():
    a = nested_lr_stack(np.column_stack([Xp] + rs), y).auc
    print(f"  pruned81 + {g:55s} ({len(rs):2d}) {a:.6f} delta {a - b_p:+.6f}", flush=True)
a = nested_lr_stack(np.column_stack([Xp] + [r for _, r in new]), y).auc
print(f"pruned81 + all new {a:.6f} delta {a - b_p:+.6f}  vs v7 {a - b_full:+.6f}", flush=True)
