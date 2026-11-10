"""Add every non-duplicate member found under DIR(s) to the cached v7 matrix; report stack delta.
    python src/pool_add_dir.py ext/ds7 [--dedup 0.9999] [--min_auc 0.93]
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from data_science_stuff.kaggle.stacking import load_oof_banks, nested_lr_stack, rank_logit

z = np.load(Path(__file__).parent.parent / "results" / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
mem = load_oof_banks(dirs, n_train=len(y), n_test=299984)
new = []
for k, (o, _) in sorted(mem.items()):
    if roc_auc_score(y, o) < 0.93:
        continue
    r = rank_logit(o)
    c = np.abs(np.corrcoef(np.column_stack([Xo[:, ::1], r]).T[-1:], np.column_stack([Xo, r]).T)[0, :-1]).max() \
        if False else max(abs(np.corrcoef(r, Xo[:, j])[0, 1]) for j in range(Xo.shape[1]))
    if c < 0.9999 and all(abs(np.corrcoef(r, q)[0, 1]) < 0.9999 for _, q in new):
        new.append((k, r)); print(f"new {k:60s} auc {roc_auc_score(y, o):.5f} maxcorr {c:.5f}", flush=True)
print(len(mem), "loaded,", len(new), "new", flush=True)
if new:
    base = 0.9526657545740411
    print("delta", f"{nested_lr_stack(np.column_stack([Xo] + [r for _, r in new]), y).auc - base:+.6f}", flush=True)
