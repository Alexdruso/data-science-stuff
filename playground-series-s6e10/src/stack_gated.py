"""Route-reliability-gated LR stack: member logits × g, g = standardised log route count
(label-free, train + test). Lets the stack shift weight between TE-driven members and
raw-column TFMs depending on how well-observed the row's route is. Nested on the base folds.

    python src/stack_gated.py --ours LEG ... [--write NAME]
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def route_gate() -> tuple[np.ndarray, np.ndarray]:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))["flight_distance"].to_numpy()
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))["flight_distance"].to_numpy()
    v, c = np.unique(np.concatenate([tr, te]), return_counts=True)
    cnt = dict(zip(v, c))
    g_tr = np.log1p(np.array([cnt[x] for x in tr], float))
    g_te = np.log1p(np.array([cnt[x] for x in te], float))
    mu, sd = g_tr.mean(), g_tr.std()
    return (g_tr - mu) / sd, (g_te - mu) / sd


def nested(y, X, T, C):
    oof, test = np.zeros(len(y)), np.zeros(len(T))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        oof[va] = lr.decision_function(sc.transform(X[va]))
        test += lr.decision_function(sc.transform(T)) / 5
    return oof, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--C", type=float, nargs="+", default=[0.1])
    p.add_argument("--write", default="")
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    _, O, T = load_pairs(a.ours, [str(DATA_DIR / "public_oof" / "_pool")])
    g, gt = route_gate()
    print(f"route-count gate: train rows with count<200: {(np.expm1(g * 0 + np.log1p(200)) > 0).sum()}", flush=True)
    variants = {
        "linear gate": (np.column_stack([O, O * g[:, None], g]), np.column_stack([T, T * gt[:, None], gt])),
        "linear+quad gate": (np.column_stack([O, O * g[:, None], O * (g**2)[:, None], g, g**2]),
                             np.column_stack([T, T * gt[:, None], T * (gt**2)[:, None], gt, gt**2])),
    }
    for name, (X, XT) in variants.items():
        for C in a.C:
            oof, test = nested(y, X, XT, C)
            print(f"{name} C={C}: {roc_auc_score(y, oof):.6f}", flush=True)
            if a.write and name == "linear gate":
                np.save(RESULTS_DIR / f"oof_{a.write}.npy", oof)
                np.save(RESULTS_DIR / f"test_{a.write}.npy", test)
                ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
                write_submission(SUBMISSIONS_DIR, f"{a.write}.csv", ids, TARGET, 1 / (1 + np.exp(-test)))


if __name__ == "__main__":
    main()
