"""Honest multi-leg rank blend (Nelder-Mead weights fit on one half, scored on the other).

    python src/blend_nm.py LEG1 LEG2 ...                  # honest score of the blend
    python src/blend_nm.py LEG1 LEG2 ... --vs REF1 REF2   # paired: blend vs reference blend
    python src/blend_nm.py LEG1 LEG2 ... --write NAME     # also write submissions/NAME.csv

Weights are non-negative, normalised, fit on a fixed random half of train and
scored on the other half, both directions (root CLAUDE.md lesson 3). The
submission uses the mean of the two half-fits. Every blend read today goes
through this one function so the halves are identical across reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def _rank(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def _fit(y: np.ndarray, R: np.ndarray) -> np.ndarray:
    k = R.shape[1]
    if k == 1:
        return np.ones(1)

    def loss(v: np.ndarray) -> float:
        w = np.abs(v)
        return -float(roc_auc_score(y, R @ (w / w.sum())))

    res = minimize(loss, np.full(k, 1.0 / k), method="Nelder-Mead",
                   options={"maxiter": 200 * k, "xatol": 1e-4, "fatol": 1e-8})
    w = np.abs(res.x)
    return w / w.sum()


def honest(y: np.ndarray, R: np.ndarray, half: np.ndarray) -> tuple[float, np.ndarray, list[float]]:
    scores, ws = [], []
    for fit_mask in (half, ~half):
        w = _fit(y[fit_mask], R[fit_mask])
        ws.append(w)
        scores.append(float(roc_auc_score(y[~fit_mask], R[~fit_mask] @ w)))
    return float(np.mean(scores)), np.mean(ws, axis=0), scores


def load(legs: list[str], split: str) -> np.ndarray:
    return np.column_stack([_rank(np.load(RESULTS_DIR / f"{split}_{leg}.npy")) for leg in legs])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("legs", nargs="+")
    p.add_argument("--vs", nargs="*", default=None)
    p.add_argument("--write", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    half = np.random.default_rng(a.seed).random(len(y)) < 0.5
    R = load(a.legs, "oof")
    for leg, col in zip(a.legs, R.T):
        print(f"  {leg:55s} solo {roc_auc_score(y, col):.6f}")
    score, w, halves = honest(y, R, half)
    print(f"BLEND honest {score:.6f}  (halves {halves[0]:.6f} / {halves[1]:.6f})  "
          f"in-sample {roc_auc_score(y, R @ w):.6f}")
    for leg, wi in zip(a.legs, w):
        print(f"    w {wi:.3f}  {leg}")
    if a.vs:
        ref, _, ref_halves = honest(y, load(a.vs, "oof"), half)
        d = [s - r for s, r in zip(halves, ref_halves)]
        print(f"REF   honest {ref:.6f}  → delta {score - ref:+.6f}  (halves {d[0]:+.6f} / {d[1]:+.6f})")
    if a.write:
        T = load(a.legs, "test")
        test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
        out = write_submission(SUBMISSIONS_DIR, f"{a.write}.csv", test_ids, TARGET, T @ w)
        print(f"Submission → {out}")


if __name__ == "__main__":
    main()
