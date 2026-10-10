"""Combiner variants over the same member set as stack_public.py (nested on the base folds).

    python src/stack_variants.py --ours LEG ... --public data/public_oof/_pool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def nested(y: np.ndarray, X: np.ndarray, C: float) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        oof[va] = lr.decision_function(sc.transform(X[va]))
    return oof


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", nargs="*", default=[])
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, _ = load_pairs(a.ours, a.public)
    base = nested(y, O, 0.1)
    print(f"base LR logits: {roc_auc_score(y, base):.6f}", flush=True)
    # Curvature: the consensus logit's square and |.|, plus member spread (disagreement)
    m = O.mean(axis=1, keepdims=True)
    extra = np.column_stack([m**2, np.abs(m), O.std(axis=1), O.std(axis=1) * m[:, 0]])
    o2 = nested(y, np.column_stack([O, extra]), 0.1)
    print(f"+ curvature/spread: {roc_auc_score(y, o2):.6f}", flush=True)
    # Rank-gauss inputs (members on a common scale, scale-free)
    R = np.column_stack([rankdata(O[:, j]) / len(O) for j in range(O.shape[1])])
    R = np.log(R / (1 - R + 1e-9) + 1e-9)
    o3 = nested(y, R, 0.1)
    print(f"rank-logit inputs: {roc_auc_score(y, o3):.6f}", flush=True)
    av = rankdata(base) + rankdata(o3)
    print(f"avg(base, rank): {roc_auc_score(y, av):.6f}", flush=True)


if __name__ == "__main__":
    main()
