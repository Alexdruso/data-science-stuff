"""Split-half combiner ranking: NM blend vs LR stack vs their average (metablend).

Same honest protocol as probe_blend_gate.py, but the arms are COMBINERS over the
same base set, not with/without a base. Per cell (6 stratified splits x 2
directions): fit each combiner + its decision weights on half A, score half B.
  nm — precorrect + Nelder-Mead blend + decision weights (today's final3 recipe)
  lr — LR on per-(model,class) logits of the raw probs + decision weights
  mb — s6e6 'metablend': uniform average of the two half-fit prob arrays,
       decision weights fit on the averaged fit-half probs (no extra fitting)
Paired deltas vs nm. Swap standard: mean holdout delta >= +0.0003, sign-consistent.

Run: S6E7_REPAIR=1 python src/probe_combiner_gate.py [key1 key2 ...]
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
SWAP_GATE = 0.0003


def _decide(
    y_fit, probs_fit, probs_eval, y_eval
) -> tuple[float, np.ndarray, np.ndarray]:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    score = balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw))
    return float(score), probs_fit, probs_eval


def cell(
    oof_pc: list[np.ndarray],
    oof_raw: list[np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[float, float, float]:
    # nm: precorrected NM blend
    fit_pc = [o[fit_idx] for o in oof_pc]
    res = minimize(
        objective,
        np.ones(len(oof_pc)),
        args=(fit_pc, y[fit_idx]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
    )
    nm_fit = blend(res.x, fit_pc)
    nm_eval = blend(res.x, [o[eval_idx] for o in oof_pc])
    nm_score, _, _ = _decide(y[fit_idx], nm_fit, nm_eval, y[eval_idx])

    # lr: logits of raw probs, one LR fit on the fit half (eval half is honest holdout)
    X_fit = np.hstack([clipped_logit(o[fit_idx]) for o in oof_raw])
    X_eval = np.hstack([clipped_logit(o[eval_idx]) for o in oof_raw])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X_fit, y[fit_idx])
    lr_fit = lr.predict_proba(X_fit)
    lr_eval = lr.predict_proba(X_eval)
    lr_score, _, _ = _decide(y[fit_idx], lr_fit, lr_eval, y[eval_idx])

    # mb: uniform average of the two combiners' probs
    mb_fit = 0.5 * (nm_fit + lr_fit)
    mb_eval = 0.5 * (nm_eval + lr_eval)
    mb_score, _, _ = _decide(y[fit_idx], mb_fit, mb_eval, y[eval_idx])
    return nm_score, lr_score, mb_score


def main() -> None:
    keys = sys.argv[1:] or DEFAULT_KEYS
    ds = load_dataset()
    y = ds.y
    oof_raw_d = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test_d = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    oof_pc_d, _ = precorrect(dict(oof_raw_d), test_d)
    oof_pc = [oof_pc_d[k] for k in keys]
    oof_raw = [oof_raw_d[k] for k in keys]

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=min(len(cells), 12))(
        delayed(cell)(oof_pc, oof_raw, y, f, e) for f, e in cells
    )
    nm = np.array([r[0] for r in results])
    lr = np.array([r[1] for r in results])
    mb = np.array([r[2] for r in results])
    print(f"combiners over {keys}")
    print(
        f"holdout weighted-bacc means: nm={nm.mean():.4f} lr={lr.mean():.4f} mb={mb.mean():.4f}"
    )
    for name, arr in [("lr-nm", lr - nm), ("mb-nm", mb - nm), ("mb-lr", mb - lr)]:
        print(
            f"  {name}: mean {arr.mean():+.5f}  sd {arr.std():.5f}  "
            f"positive {int((arr > 0).sum())}/{len(arr)}  "
            f"cells {np.round(arr, 5).tolist()}"
        )
    best = max([("lr", (lr - nm).mean()), ("mb", (mb - nm).mean())], key=lambda t: t[1])
    verdict = f"{best[0]} beats nm by {best[1]:+.5f} -> " + (
        "SWAP-WORTHY" if best[1] >= SWAP_GATE else "below swap gate"
    )
    print(f"VERDICT: {verdict} (swap gate +{SWAP_GATE})")


if __name__ == "__main__":
    main()
