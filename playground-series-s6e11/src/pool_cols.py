"""Stack delta of ad-hoc column groups added to the cached v7 matrix (+ megayak LM = pool_v11 base).
    python src/pool_cols.py   (edit GROUPS)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from data_science_stuff.kaggle.stacking import nested_lr_stack, rank_logit

R = Path(__file__).parent.parent
z = np.load(R / "results" / "pool_lab.npz"); Xo, y = z["Xo"], z["y"]
mega = rank_logit(np.load(R / "ext/ds23/megayak-lm/lm_oof.npy"))
V = R / "ext/ds29/s6e11-sovereign-oof-test-and-orthogonal-vault"
s24 = rank_logit(np.load(V / "oof_s24_chimera_bpe_synergy.npy"))
tri = rank_logit(np.load(V / "oof_trigen_native_logodds_hybrid.npy"))
ro = np.load(V / "orthogonal_control_variates_oof_test.npz")["oof_r_orth"]
ve = np.load(V / "order_statistic_7way_vetoes_oof_test.npz")["oof_vetoes"]
for j in range(ro.shape[1]): print("r_orth", j, roc_auc_score(y, ro[:, j]), ro[:, j].std())
for j in range(ve.shape[1]): print("veto", j, roc_auc_score(y, ve[:, j]), ve[:, j].std(), len(np.unique(ve[:, j])))
B = np.column_stack([Xo, mega])
base = nested_lr_stack(B, y, seeds=(7,)).auc
print("base v7+mega (seed 7)", base, flush=True)
G = {"s24": s24[:, None], "trigen": tri[:, None], "r_orth": np.column_stack([rank_logit(c) for c in ro.T]),
     "vetoes": np.column_stack([rank_logit(c) for c in ve.T])}
for k, a in G.items():
    print(k, f"{nested_lr_stack(np.column_stack([B, a]), y, seeds=(7,)).auc - base:+.6f}", flush=True)
