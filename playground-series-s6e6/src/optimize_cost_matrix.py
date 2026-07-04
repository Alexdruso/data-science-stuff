"""Full 3x3 cost-matrix decision rule on stack OOFs — generalized threshold opt (S6E6).

The incumbent per-class scaling argmax(w_k * P_k) (postprocess.optimize_thresholds) is
the C[j,k] = a_j special case of the Bayes decision rule argmin_k sum_j C[j,k] P(j|x):
a per-class scale cannot penalize a SPECIFIC confusion pair. Tonight's EDA: GAL->STAR
alone is 9,449 errors (~48% of all confusion) — a pair-targeted cost can shift only
that boundary while leaving GAL->QSO untouched.

6 off-diagonal costs, scale-invariant => 5 effective params on 577k rows. Honest gate:
split-half — fit C on half the OOF rows, score on the held-out half, against per-class
scaling fitted the SAME way (3 seeds x both halves = 6 paired measurements). Only if
the mean holdout delta >= 0 do we fit on the full OOF and emit a {run}_costmx
submission (never overwrites the incumbent {run}.csv).

Run:  python src/optimize_cost_matrix.py [--runs lrstack gbdtstack]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from postprocess import optimize_thresholds

from data_science_stuff.kaggle.decision import (
    cost_decide,
    fit_cost_matrix,
    make_cost_matrix,
    split_half_gate,
)
from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_CLASSES = 3

# The cost-matrix machinery (make_cost_matrix / cost_decide / fit_cost_matrix /
# split_half_gate) lives in data_science_stuff.kaggle.decision, generalized to
# n classes; these wrappers pin the s6e6 3-class shape and keep the original
# local names.


def make_cost(log_c: np.ndarray) -> np.ndarray:
    return make_cost_matrix(log_c, N_CLASSES)


def fit_cost(
    proba: np.ndarray,
    y: np.ndarray,
    warm_weights: np.ndarray,
    n_restarts: int = 8,
) -> tuple[np.ndarray, float]:
    return fit_cost_matrix(proba, y, warm_weights, n_restarts=n_restarts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["lrstack", "gbdtstack"])
    args = parser.parse_args()

    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    for run in args.runs:
        oof = np.load(RESULTS_DIR / f"oof_{run}.npy")
        test = np.load(RESULTS_DIR / f"test_{run}.npy")
        print(f"\n===== {run} =====")

        w_full, pc_full = optimize_thresholds(oof, y)
        print(f"incumbent per-class scaling (full OOF, in-sample): {pc_full:.5f}")

        print("split-half gate (fit on one half, score on the other):")
        deltas = split_half_gate(oof, y, n_seeds=3, n_restarts=5)
        print(f"  per-split deltas: {[round(d, 5) for d in deltas]}")
        mean_delta = float(np.mean(deltas))
        print(f"mean holdout delta (cost-mx - per-class): {mean_delta:+.5f}")

        log_c, cm_full = fit_cost(oof, y, w_full)
        cost = make_cost(log_c)
        pred = cost_decide(oof, cost)
        rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
        print(f"full-OOF cost-matrix score: {cm_full:.5f} (vs per-class {pc_full:.5f})")
        print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")
        print("cost matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cost, index=le.classes_, columns=le.classes_).round(3))

        out = {
            "run": run,
            "classes": le.classes_.tolist(),
            "cost_matrix": cost.tolist(),
            "full_oof_score": cm_full,
            "per_class_scaling_score": pc_full,
            "mean_holdout_delta": mean_delta,
        }
        with (RESULTS_DIR / f"cost_matrix_{run}.json").open("w") as f:
            json.dump(out, f, indent=2)

        if mean_delta >= 0 and cm_full > pc_full:
            labels = le.inverse_transform(cost_decide(test, cost))
            sub = write_submission(SUBMISSIONS_DIR, f"{run}_costmx.csv", test_ids, TARGET, labels)
            print(f"GATE PASSED -> saved {sub.name}")
        else:
            print("GATE FAILED (holdout delta < 0 or no full-OOF gain) -> no submission")


if __name__ == "__main__":
    main()
