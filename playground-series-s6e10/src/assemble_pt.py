"""Assemble per-fold _pt_<tag>/f{k}_{val,test}.npy (from kind_leg.py / rf_leg.py) into a full
results/oof_<name>.npy (id order) + test_<name>.npy (mean of the 5 fold models).

    python src/assemble_pt.py linear mono xrank rf_l30_f0.33 et_l20_f0.5
"""

import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_TR = 699_635

for tag in sys.argv[1:]:
    d = RESULTS / f"_pt_{tag}"
    oof, seen, tests = np.full(N_TR, np.nan), np.zeros(N_TR, bool), []
    for k in range(1, 6):
        idx, pv = np.load(d / f"f{k}_val.npy")
        oof[idx.astype(int)] = pv
        seen[idx.astype(int)] = True
        tests.append(np.load(d / f"f{k}_test.npy"))
    assert seen.all(), tag
    np.save(RESULTS / f"oof_kind_{tag}.npy", oof)
    np.save(RESULTS / f"test_kind_{tag}.npy", np.mean(tests, axis=0))
    print(f"kind_{tag}: ok")
