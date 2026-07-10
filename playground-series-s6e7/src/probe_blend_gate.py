"""Split-half blend gate: does adding a candidate base to the blend survive holdout?

The blend-overfit failure mode (Day-5): NM hands a near-tie leg a big weight for
+0.0001 in-sample. This gate fits everything (NM blend weights + decision
weights) on one half of the OOF rows and scores the OTHER half, with and
without the candidate, over N stratified splits x both directions. Deltas are
paired per cell. PASS = mean holdout delta >= +0.0005 (plan gate).

Usage:
  python src/probe_blend_gate.py <candidate_key> <base_key1> <base_key2> ...
  e.g. probe_blend_gate.py realmlp_r_breadth lgbm_r_breadth hgbc_r_breadth \
       xgboost_r_breadth catboost_r_breadth
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import RESULTS_DIR, blend, objective, precorrect
from train_common import load_dataset

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
GATE = 0.0005


def fit_eval(
    oof_list: list[np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> float:
    """Fit NM blend + decision weights on fit_idx, return weighted bacc on eval_idx."""
    fit_probs = [o[fit_idx] for o in oof_list]
    res = minimize(
        objective,
        np.ones(len(oof_list)),
        args=(fit_probs, y[fit_idx]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
    )
    blended_fit = blend(res.x, fit_probs)
    dw = np.asarray(tune_decision_weights(y[fit_idx], blended_fit))
    blended_eval = blend(res.x, [o[eval_idx] for o in oof_list])
    return float(
        balanced_accuracy_score(y[eval_idx], weighted_predict(blended_eval, dw))
    )


def cell(
    with_c: list[np.ndarray],
    without_c: list[np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[float, float]:
    return (
        fit_eval(with_c, y, fit_idx, eval_idx),
        fit_eval(without_c, y, fit_idx, eval_idx),
    )


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: probe_blend_gate.py <candidate> <base1> <base2> ...")
    cand, bases = sys.argv[1], sys.argv[2:]
    keys = bases + [cand]
    ds = load_dataset()
    y = ds.y

    oof = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys}
    test = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys}
    oof, _ = precorrect(oof, test)
    with_c = [oof[k] for k in keys]
    without_c = [oof[k] for k in bases]

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=min(len(cells), 12))(
        delayed(cell)(with_c, without_c, y, f, e) for f, e in cells
    )
    deltas = np.array([w - wo for w, wo in results])
    print(f"candidate {cand} over bases {bases}")
    print(
        f"holdout weighted-bacc: with={np.mean([r[0] for r in results]):.4f} "
        f"without={np.mean([r[1] for r in results]):.4f}"
    )
    print(f"paired deltas per cell: {np.round(deltas, 5).tolist()}")
    print(
        f"mean {deltas.mean():+.5f}  sd {deltas.std():.5f}  "
        f"positive {int((deltas > 0).sum())}/{len(deltas)}"
    )
    verdict = "PASS" if deltas.mean() >= GATE else "FAIL"
    print(f"GATE (mean holdout delta >= +{GATE}): {verdict}")


if __name__ == "__main__":
    main()
