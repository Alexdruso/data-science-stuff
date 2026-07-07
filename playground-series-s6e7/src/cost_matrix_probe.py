"""Full 3x3 cost-matrix decision layer on the champion blend, honestly gated (plan #1).

The 07-06 kill-audit un-killed this: per-class weights argmax(w_c * p_c) are the
special case C[j,k] = w_j of the Bayes rule argmin_k sum_j p_j C[j,k]; the full
matrix (6 free off-diagonal params) can tilt the at-risk<->unhealthy boundary
independently of at-risk<->fit -- where 79% of the v1b/v2 flips lived.

In-sample NM on 6 params WILL overfit, so the verdict comes exclusively from
data_science_stuff.kaggle.decision.split_half_gate: both rules fit on one half
of the OOF rows, scored on the other, over seed/half swaps.
GATE: mean holdout delta >= +0.001 (the LB-blind protocol's minimum effect size).

The champion (v1b) blended OOF is reconstructed exactly as forensics_v1b_v2.py
does: _v1 legs, single-shot per-leg decision weights, precorrect, NM blend refit.

Run: ../.venv/bin/python src/cost_matrix_probe.py | tee results/cost_matrix.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from lb_anchor import correct
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.decision import (
    cost_decide,
    fit_cost_matrix,
    make_cost_matrix,
    optimize_thresholds,
    split_half_gate,
)
from data_science_stuff.kaggle_utils import tune_decision_weights

GBDTS = ["lgbm", "xgboost", "catboost"]
GATE = 0.001


def champion_blend_oof(y: NDArray[np.int64]) -> NDArray[np.float64]:
    """Reproduce the v1b blended OOF from the _v1 legs (forensics recipe)."""
    oof = {m: np.load(RESULTS_DIR / f"oof_{m}_v1.npy") for m in GBDTS}
    w = {m: tune_decision_weights(y, oof[m]) for m in GBDTS}
    pc_oof = [correct(oof[m], w[m]) for m in GBDTS]

    def blend(wv: NDArray, arrays: list[NDArray]) -> NDArray:
        s = np.exp(wv) / np.exp(wv).sum()
        return sum(s[i] * a for i, a in enumerate(arrays))

    res = minimize(
        lambda wv: (
            -float(balanced_accuracy_score(y, np.argmax(blend(wv, pc_oof), axis=1)))
        ),
        np.ones(len(GBDTS)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    print(f"blend weights: {np.round(np.exp(res.x) / np.exp(res.x).sum(), 4)}")
    return blend(res.x, pc_oof)


def main() -> None:
    ds = load_dataset()
    proba = champion_blend_oof(ds.y)

    w, per_class_score = optimize_thresholds(proba, ds.y)
    print(f"per-class weights: {w.round(4)}  in-sample bacc {per_class_score:.4f}")

    log_c, cost_score = fit_cost_matrix(proba, ds.y, w)
    cost = make_cost_matrix(log_c, proba.shape[1])
    print(f"cost matrix (rows=truth, cols=pred):\n{cost.round(4)}")
    print(
        f"in-sample bacc: cost-matrix {cost_score:.4f} vs per-class "
        f"{per_class_score:.4f} (delta {cost_score - per_class_score:+.4f}) "
        "-- in-sample only, the verdict is below"
    )
    in_sample_pred_shift = float(
        (cost_decide(proba, cost) != np.argmax(proba * w, axis=1)).mean()
    )
    print(f"rows moved vs per-class rule (in-sample): {in_sample_pred_shift:.3%}")

    print("\nsplit-half honest gate (fit one half, score the other):")
    deltas = split_half_gate(proba, ds.y)
    for i, d in enumerate(deltas):
        print(f"  holdout delta {i + 1}: {d:+.5f}")
    mean_d = float(np.mean(deltas))
    verdict = "PASS" if mean_d >= GATE else "FAIL"
    print(
        f"\nGATE (mean holdout delta, need >=+{GATE}): {mean_d:+.5f} "
        f"(sd {np.std(deltas):.5f}) -> {verdict}"
    )


if __name__ == "__main__":
    main()
