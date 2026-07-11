"""Day-9 combiner tournament: do the new legs earn LR-stack seats?

Same honest split-half protocol as probe_combiner_gate.py (6 stratified splits
x 2 directions; every combiner + its decision weights fit on half A, scored on
half B), leaner arm set focused on ONE question: does adding any Day-9 leg
(cascade_r_s42 / tabms_r_s42 / realmlp_rule_r_s42) to the incumbent lrstack6
base set displace it? gbdtmeta/nnmeta dropped (lost decisively on Day-8).

Arms:
  nm     — 5-base precorrect + Nelder-Mead (final3 recipe, anchor)
  lr6    — INCUMBENT: strong-5 + mlp_la_r logits -> LR (the lrstack6_r engine)
  lr6c   — lr6 + cascade_r_s42
  lr6t   — lr6 + tabms_r_s42
  lr6ct  — lr6 + cascade + tabms
  lr9    — lr6 + cascade + tabms + realmlp_rule_r_s42
  mb_<a> — avg(<a>, nm) + decision weights (metablend shape) for each lr arm

Displacement gate (vs lr6, NOT vs nm): mean holdout delta >= +0.0003 and >= 10/12
cells positive.

Run: S6E7_REPAIR=1 python src/probe_combiner_day9.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from build_lr_stack import DEFAULT_KEYS
from ensemble import RESULTS_DIR, blend, objective, precorrect
from train_common import load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
DISPLACE_GATE = 0.0003
LR6_EXTRA = ["mlp_la_r"]
ARM_EXTRAS: dict[str, list[str]] = {
    "lr6": [],
    "lr6c": ["cascade_r_s42"],
    "lr6t": ["tabms_r_s42"],
    "lr6ct": ["cascade_r_s42", "tabms_r_s42"],
    "lr9": ["cascade_r_s42", "tabms_r_s42", "realmlp_rule_r_s42"],
}


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _fit_lr(blocks, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    X_fit = np.hstack([b[fit_idx] for b in blocks])
    X_eval = np.hstack([b[eval_idx] for b in blocks])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X_fit, y[fit_idx])
    return lr.predict_proba(X_fit), lr.predict_proba(X_eval)


def cell(
    oof_pc: list[np.ndarray],
    arm_blocks: dict[str, list[np.ndarray]],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> dict[str, float]:
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_pc = [o[fit_idx] for o in oof_pc]
    res = minimize(
        objective,
        np.ones(len(oof_pc)),
        args=(fit_pc, y[fit_idx]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
    )
    pairs["nm"] = (blend(res.x, fit_pc), blend(res.x, [o[eval_idx] for o in oof_pc]))
    for name, blocks in arm_blocks.items():
        pairs[name] = _fit_lr(blocks, y, fit_idx, eval_idx)
    for name in arm_blocks:
        pairs[f"mb_{name}"] = (
            np.mean([pairs[name][0], pairs["nm"][0]], axis=0),
            np.mean([pairs[name][1], pairs["nm"][1]], axis=0),
        )
    return {
        name: _decide(y[fit_idx], pf, pe, y[eval_idx])
        for name, (pf, pe) in pairs.items()
    }


def main() -> None:
    keys = DEFAULT_KEYS
    all_keys = sorted(
        set(keys + LR6_EXTRA + [k for ex in ARM_EXTRAS.values() for k in ex])
    )
    ds = load_dataset()
    y = ds.y
    oof_raw = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in all_keys}
    test_raw = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in all_keys}
    oof_pc_d, _ = precorrect(
        {k: oof_raw[k] for k in keys}, {k: test_raw[k] for k in keys}
    )
    oof_pc = [oof_pc_d[k] for k in keys]
    logit_d = {k: clipped_logit(oof_raw[k]) for k in all_keys}
    base6 = [logit_d[k] for k in keys + LR6_EXTRA]
    arm_blocks = {
        name: base6 + [logit_d[k] for k in extras]
        for name, extras in ARM_EXTRAS.items()
    }

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(oof_pc, arm_blocks, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    print(f"Day-9 combiner tournament; {len(cells)} cells; incumbent = lr6")
    print("mean holdout weighted-bacc per arm (deltas vs lr6):")
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr6"]
        print(
            f"  {a:<9} {scores[a].mean():.5f}   vs lr6 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    challengers = [a for a in arms if a not in ("lr6", "nm")]
    best = max(challengers, key=lambda a: scores[a].mean())
    d = scores[best] - scores["lr6"]
    verdict = (
        "DISPLACES lrstack6_r"
        if d.mean() >= DISPLACE_GATE and (d > 0).sum() >= 10
        else "below displacement gate — final #2 stays lrstack6_r"
    )
    print(f"\nBEST CHALLENGER: {best}  vs lr6 {d.mean():+.5f} "
          f"({int((d > 0).sum())}/{len(d)})  -> {verdict}")


if __name__ == "__main__":
    main()
