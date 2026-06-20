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
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from postprocess import optimize_thresholds

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
# (j, k) pairs: cost C[j, k] of predicting k when the true class is j.
OFF_DIAG: list[tuple[int, int]] = [(j, k) for j in range(3) for k in range(3) if j != k]
N_SPLIT_SEEDS = 3


def make_cost(log_c: np.ndarray) -> np.ndarray:
    cost = np.zeros((3, 3))
    for (j, k), v in zip(OFF_DIAG, np.exp(log_c)):
        cost[j, k] = v
    return cost


def cost_decide(proba: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Bayes rule: predict argmin_k of expected cost sum_j P(j|x) * C[j, k]."""
    return np.argmin(proba @ cost, axis=1)


def fit_cost(
    proba: np.ndarray,
    y: np.ndarray,
    warm_weights: np.ndarray,
    n_restarts: int = 8,
) -> tuple[np.ndarray, float]:
    """Nelder-Mead over the 6 log-costs; warm-started from the per-class optimum.

    Per-class scaling argmax(w_k P_k) == cost matrix C[j, k] = w_j (k != j), so the
    warm start guarantees the in-sample optimum is >= the per-class one.
    """

    def neg_ba(log_c: np.ndarray) -> float:
        pred = cost_decide(proba, make_cost(log_c))
        return -float(balanced_accuracy_score(y, pred))

    warm = np.array([np.log(warm_weights[j]) for j, _ in OFF_DIAG])
    best_x, best_score = warm, -neg_ba(warm)
    starts = [warm] + [
        warm + np.random.default_rng(seed).normal(0, 0.3, size=6)
        for seed in range(n_restarts)
    ]
    for x0 in starts:
        result = minimize(
            neg_ba, x0, method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if -result.fun > best_score:
            best_score, best_x = -result.fun, result.x
    return best_x, best_score


def split_half_gate(proba: np.ndarray, y: np.ndarray) -> float:
    """Mean holdout delta (cost-matrix - per-class) over 3 seeds x 2 half-swaps."""
    deltas = []
    for seed in range(N_SPLIT_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        halves = (idx[: len(y) // 2], idx[len(y) // 2 :])
        for fit_idx, val_idx in (halves, halves[::-1]):
            w, _ = optimize_thresholds(proba[fit_idx], y[fit_idx], n_restarts=5)
            pc_val = balanced_accuracy_score(
                y[val_idx], np.argmax(proba[val_idx] * w, axis=1)
            )
            log_c, _ = fit_cost(proba[fit_idx], y[fit_idx], w, n_restarts=5)
            cm_val = balanced_accuracy_score(
                y[val_idx], cost_decide(proba[val_idx], make_cost(log_c))
            )
            deltas.append(cm_val - pc_val)
            print(
                f"  seed {seed} half: per-class {pc_val:.5f}  cost-mx {cm_val:.5f}  "
                f"delta {cm_val - pc_val:+.5f}"
            )
    return float(np.mean(deltas))


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
        mean_delta = split_half_gate(oof, y)
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
            SUBMISSIONS_DIR.mkdir(exist_ok=True)
            sub = SUBMISSIONS_DIR / f"{run}_costmx.csv"
            pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(sub, index=False)
            print(f"GATE PASSED -> saved {sub.name}")
        else:
            print("GATE FAILED (holdout delta < 0 or no full-OOF gain) -> no submission")


if __name__ == "__main__":
    main()
