"""PR-36c: bagged deep-tree families (RandomForest / ExtraTrees) on the exported fold-safe matrices.

    python src/rf_leg.py --model rf --folds 1        # results/_pt_<model>/f{k}_{val,test}.npy

Absent from the public pool (every tree member there is boosted). Fully out-of-fold: the forest
is fit on the fold's training rows only, no early stopping.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rf", "et"], default="rf")
    p.add_argument("--folds", type=int, nargs="+", default=[1])
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--leaf", type=int, default=30)
    p.add_argument("--maxf", type=float, default=0.33)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    tag = f"{a.model}_l{a.leaf}_f{a.maxf}"
    out = RESULTS / f"_pt_{tag}"
    out.mkdir(exist_ok=True)
    cls = RandomForestClassifier if a.model == "rf" else ExtraTreesClassifier
    for k in a.folds:
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        med = np.nanmedian(z["x_tr"], axis=0)
        for key in ("x_tr", "x_va", "x_te"):
            x = z[key].astype(np.float32)
            r, c = np.where(np.isnan(x))
            x[r, c] = med[c]
            z[key] = x
        t0 = time.time()
        m = cls(n_estimators=a.trees, min_samples_leaf=a.leaf, max_features=a.maxf, n_jobs=a.jobs,
                random_state=42 + k).fit(z["x_tr"], z["y_tr"].astype(int))
        pv = m.predict_proba(z["x_va"])[:, 1]
        print(f"fold {k}: AUC {roc_auc_score(z['y_va'], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), pv]))
        np.save(out / f"f{k}_test.npy", m.predict_proba(z["x_te"])[:, 1])
        del m


if __name__ == "__main__":
    main()
