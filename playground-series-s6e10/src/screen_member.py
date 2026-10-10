"""Cheap incremental-value screen of a candidate member against the current stack.

    python src/screen_member.py --stack hybrid_lrstack_1010 --probe results/probe_<tag>_f1.npy

The probe holds (row index, prediction) for a subset of one validation fold. On
those rows, a 2-feature LR (stack logit, candidate logit) is cross-validated
(10-fold, repeated) and compared with the stack logit alone. The stack's OOF is
nested, so both arms are honest; the delta bounds what a full run can add.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def _logit(p: np.ndarray) -> np.ndarray:
    if p.min() < 0 or p.max() > 1:
        return p
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="hybrid_lrstack_1010")
    ap.add_argument("--probe", nargs="+", required=True)
    a = ap.parse_args()
    y_all = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    s_all = _logit(np.load(RESULTS_DIR / f"oof_{a.stack}.npy"))
    for path in a.probe:
        idx_f, pv = np.load(path)
        idx = idx_f.astype(int)
        y, s, c = y_all[idx], s_all[idx], _logit(pv)
        base, both = [], []
        for rep in range(5):
            o1, o2 = np.zeros(len(y)), np.zeros(len(y))
            for tr, va in StratifiedKFold(10, shuffle=True, random_state=rep).split(s, y):
                X = np.column_stack([s, c])
                o2[va] = LogisticRegression().fit(X[tr], y[tr]).decision_function(X[va])
                o1[va] = s[va]
            base.append(roc_auc_score(y, o1))
            both.append(roc_auc_score(y, o2))
        d = np.array(both) - np.array(base)
        corr = np.corrcoef(s, c)[0, 1]
        print(f"{Path(path).stem}: n={len(y)} solo {roc_auc_score(y, c):.5f} stack {base[0]:.5f} "
              f"corr {corr:.4f}  delta {d.mean():+.6f} ± {d.std():.6f}")


if __name__ == "__main__":
    main()
