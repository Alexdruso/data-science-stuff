"""Split-half combiner tournament: NM / LR / LR6 / ridgecal-GBDT / NN metas + averages.

Same honest protocol as probe_blend_gate.py, generalized to N arms. Per cell
(6 stratified splits x 2 directions): fit EVERY combiner + its decision weights
on half A, score half B. Uniform averages of arm outputs are computed post-hoc
inside the cell (no extra fitting — the s6e6 'metablend' move).

Arms:
  nm       — precorrect + Nelder-Mead blend (the final3 recipe)
  lr       — LR on per-(model,class) logits, strong-5 bases
  lr6      — lr + mlp_la_r as 6th base (Caruana hinted a per-class seat)
  gbdtmeta — per-base ridge 3->3 calibration -> regularized LGBM meta (s6e6 altmeta)
  nnmeta   — small MLP meta over standardized logits, inv-freq oversampled
  mb_*     — uniform averages: lr+nm (current final #2 recipe), lr+gbdtmeta
             (the literal s6e6 best-LB pairing), lr+nm+gbdtmeta
Paired deltas vs nm. Swap standard: mean holdout delta and sign-consistency.

Run: S6E7_REPAIR=1 python src/probe_combiner_gate.py
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
from build_alt_meta import fit_gbdtmeta, fit_nnmeta
from build_lr_stack import DEFAULT_KEYS
from ensemble import RESULTS_DIR, blend, objective, precorrect
from train_common import load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
SWAP_GATE = 0.0003
EXTRA_KEY = "mlp_la_r"  # lr6's sixth base
AVERAGES = {
    "mb_lr_nm": ("lr", "nm"),
    "mb_lr_gbdt": ("lr", "gbdtmeta"),
    "mb3": ("lr", "nm", "gbdtmeta"),
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
    blocks5: list[np.ndarray],
    blocks6: list[np.ndarray],
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
    pairs["lr"] = _fit_lr(blocks5, y, fit_idx, eval_idx)
    pairs["lr6"] = _fit_lr(blocks6, y, fit_idx, eval_idx)
    pairs["gbdtmeta"] = fit_gbdtmeta(blocks5, y, fit_idx, eval_idx)
    pairs["nnmeta"] = fit_nnmeta(blocks5, y, fit_idx, eval_idx)
    for name, members in AVERAGES.items():
        pairs[name] = (
            np.mean([pairs[m][0] for m in members], axis=0),
            np.mean([pairs[m][1] for m in members], axis=0),
        )
    return {
        name: _decide(y[fit_idx], pf, pe, y[eval_idx])
        for name, (pf, pe) in pairs.items()
    }


def main() -> None:
    keys = DEFAULT_KEYS
    ds = load_dataset()
    y = ds.y
    oof_raw_d = {k: np.load(RESULTS_DIR / f"oof_{k}.npy") for k in keys + [EXTRA_KEY]}
    test_d = {k: np.load(RESULTS_DIR / f"test_{k}.npy") for k in keys + [EXTRA_KEY]}
    oof_pc_d, _ = precorrect(
        {k: oof_raw_d[k] for k in keys}, {k: test_d[k] for k in keys}
    )
    oof_pc = [oof_pc_d[k] for k in keys]
    blocks5 = [clipped_logit(oof_raw_d[k]) for k in keys]
    blocks6 = blocks5 + [clipped_logit(oof_raw_d[EXTRA_KEY])]

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(oof_pc, blocks5, blocks6, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    print(f"combiner tournament over {keys} (+{EXTRA_KEY} in lr6); {len(cells)} cells")
    print("mean holdout weighted-bacc per arm:")
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["nm"]
        print(
            f"  {a:<11} {scores[a].mean():.5f}   vs nm {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    best = max((a for a in arms if a != "nm"), key=lambda a: scores[a].mean())
    d = scores[best] - scores["nm"]
    print(
        f"\nBEST: {best}  vs nm {d.mean():+.5f}  "
        f"{'SWAP-WORTHY' if d.mean() >= SWAP_GATE else 'below swap gate'} "
        f"(gate +{SWAP_GATE})"
    )


if __name__ == "__main__":
    main()
