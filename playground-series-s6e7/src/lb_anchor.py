"""LB-anchored offline metric validation for PS S6E7.

We hold 5 scored submissions (ground-truth LB). Plain OOF is saturated and mis-ranked them
(v2 best-OOF/worst-LB). This script reconstructs each scored config's OOF prediction vector
and asks: WHICH offline metric reproduces the known LB ordering? The winner becomes the
selection rule for tomorrow's submissions.

Known LB anchors (2026-07-02):
    lgbm solo .94886 | xgb solo .94894 | ens_v1(raw-argmax) .94861 | ens_v1b .94970
    | ens_v2(bag+mlp) .94910

Caveat: reconstructed configs are retrained, not bit-identical to the scored ones — good
enough for metric ranking, not for exact score matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict
from train_common import DATA_DIR, RESULTS_DIR, load_dataset, robust_decision_weights

KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
LB = {
    "lgbm_solo": 0.94886,
    "xgb_solo": 0.94894,
    "ens_v1_raw": 0.94861,
    "ens_v1b": 0.94970,
    "ens_v2_bagmlp": 0.94910,
}


def nm_blend(oofs: list[NDArray], y: NDArray, precorrected: bool) -> NDArray:
    """Nelder-Mead softmax blend maximizing argmax balanced-acc (ensemble.py recipe)."""

    def blend(w: NDArray) -> NDArray:
        s = np.exp(w) / np.exp(w).sum()
        return sum(s[i] * a for i, a in enumerate(oofs))

    def neg(w: NDArray) -> float:
        return -float(balanced_accuracy_score(y, np.argmax(blend(w), axis=1)))

    res = minimize(
        neg,
        np.ones(len(oofs)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    return blend(res.x)


def correct(a: NDArray, w: NDArray) -> NDArray:
    p = a * w
    return p / p.sum(axis=1, keepdims=True)


def build_config_preds(y: NDArray) -> dict[str, NDArray]:
    """Reconstruct the 5 scored configs' OOF label predictions."""
    v1 = {
        m: np.load(RESULTS_DIR / f"oof_{m}_v1.npy")
        for m in ["lgbm", "xgboost", "catboost"]
    }
    bag = {
        m: np.load(RESULTS_DIR / f"oof_{m}_bag.npy")
        for m in ["lgbm", "xgboost", "catboost", "mlp"]
    }

    preds: dict[str, NDArray] = {}
    w_by_model = {m: tune_decision_weights(y, v1[m]) for m in v1}

    preds["lgbm_solo"] = weighted_predict(v1["lgbm"], w_by_model["lgbm"])
    preds["xgb_solo"] = weighted_predict(v1["xgboost"], w_by_model["xgboost"])

    # v1: blend RAW probs under argmax objective, then decision-tune (the collapsed config)
    raw_blend = nm_blend(list(v1.values()), y, precorrected=False)
    preds["ens_v1_raw"] = weighted_predict(
        raw_blend, tune_decision_weights(y, raw_blend)
    )

    # v1b (champion): precorrect each model by its own weights, blend, decision-tune
    pc = [correct(v1[m], w_by_model[m]) for m in v1]
    v1b_blend = nm_blend(pc, y, precorrected=True)
    preds["ens_v1b"] = weighted_predict(v1b_blend, tune_decision_weights(y, v1b_blend))

    # v2: bagged arrays + mlp, precorrect by robust weights, blend, robust decision-tune
    rw = {m: robust_decision_weights(y, bag[m]) for m in bag}
    pc2 = [correct(bag[m], rw[m]) for m in bag]
    v2_blend = nm_blend(pc2, y, precorrected=True)
    preds["ens_v2_bagmlp"] = weighted_predict(
        v2_blend, robust_decision_weights(y, v2_blend)
    )
    return preds


def weighted_bacc(y: NDArray, pred: NDArray, sw: NDArray) -> float:
    return float(
        np.mean([np.average(pred[y == c] == c, weights=sw[y == c]) for c in range(3)])
    )


def main() -> None:
    ds = load_dataset()
    y = ds.y

    # Adversarial train-row scores (cache; computed over the FULL feature set incl. water)
    adv_path = RESULTS_DIR / "adv_scores_train.npy"
    if adv_path.exists():
        adv = np.load(adv_path)
    else:
        from adv_eval import adversarial_scores

        adv = adversarial_scores(ds)
        np.save(adv_path, adv)

    miss_key = (
        pl.read_csv(DATA_DIR / "train.csv")
        .sort("id")
        .select(pl.any_horizontal([pl.col(c).is_null() for c in KEY_DRIVERS]))
        .to_numpy()
        .ravel()
    )

    preds = build_config_preds(y)

    metrics: dict[str, dict[str, float]] = {}
    for name, pred in preds.items():
        row = {"plain": float(balanced_accuracy_score(y, pred))}
        for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = adv >= np.quantile(adv, q)
            row[f"testlike_q{int(q * 100)}"] = float(
                balanced_accuracy_score(y[mask], pred[mask])
            )
        row["advwt"] = weighted_bacc(y, pred, adv)
        tl = adv >= np.quantile(adv, 0.7)
        row["testlike70_missdrv"] = float(
            balanced_accuracy_score(y[tl & miss_key], pred[tl & miss_key])
        )
        metrics[name] = row

    names = list(LB)
    lb_vals = [LB[n] for n in names]
    print(
        f"\n{'config':<15} LB      " + " ".join(f"{k:>16}" for k in metrics[names[0]])
    )
    for n in names:
        print(
            f"{n:<15} {LB[n]:.5f} "
            + " ".join(f"{metrics[n][k]:>16.5f}" for k in metrics[n])
        )

    print(f"\n{'metric':<20} {'kendall_tau':>11} {'v1b>v2':>7} {'xgb>lgbm':>9}")
    report = []
    for k in metrics[names[0]]:
        vals = [metrics[n][k] for n in names]
        tau = kendalltau(vals, lb_vals).statistic
        ok1 = metrics["ens_v1b"][k] > metrics["ens_v2_bagmlp"][k]
        ok2 = metrics["xgb_solo"][k] > metrics["lgbm_solo"][k]
        report.append((k, tau, ok1, ok2))
        print(f"{k:<20} {tau:>11.3f} {str(ok1):>7} {str(ok2):>9}")

    best = max(report, key=lambda r: (r[1], r[2], r[3]))
    print(
        f"\nBEST OFFLINE RANKER: {best[0]} (tau={best[1]:.3f}, v1b>v2={best[2]}, xgb>lgbm={best[3]})"
    )
    out = RESULTS_DIR / "lb_anchor_report.txt"
    with out.open("w") as f:
        f.write(f"{report}\nbest={best[0]}\n")
    print(f"report saved → {out}")


if __name__ == "__main__":
    main()
