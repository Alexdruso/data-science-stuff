"""Honest blend read for a candidate leg against the GBDT control blend (PR-2).

    python src/blend_check.py realmlp_v1
    python src/blend_check.py realmlp_e8 --base lgbm_pr5_tuned --vs realmlp_v1   # PR-7
    python src/blend_check.py tabm_v1 --control-blend   # PR-8: vs the submitted blend

control   = equal rank-average of CONTROL legs
candidate = (1 - w) * control + w * rank(leg), w on a 0.05 grid
`w` is fit on a random half of train and scored on the other half, both
directions; the reported delta is the mean of the two held-out deltas. Fitting
`w` in-sample would hand the candidate a free optimisation the control never
gets (root CLAUDE.md lesson 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

CONTROL = ["lgbm_pr1_base", "xgb_v1", "catboost_v1"]
# The submitted blend (2026-10-05): 0.65·rank(tuned LGBM) + 0.35·rank(RealMLP).
SUBMITTED_BLEND = {"lgbm_pr5_tuned": 0.65, "realmlp_v1": 0.35}
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BAR = 0.0005


def _rank(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def honest_blend_score(
    y: np.ndarray, base: np.ndarray, cand: np.ndarray, half: np.ndarray
) -> tuple[float, list[float]]:
    """Mean held-out AUC of (1-w)·base + w·cand, w fit on the other half."""
    scores, ws = [], []
    for fit_mask in (half, ~half):
        score_mask = ~fit_mask
        w = max(GRID, key=lambda g: roc_auc_score(
            y[fit_mask], (1 - g) * base[fit_mask] + g * cand[fit_mask]))
        ws.append(float(w))
        scores.append(float(roc_auc_score(
            y[score_mask], (1 - w) * base[score_mask] + w * cand[score_mask])))
    return float(np.mean(scores)), ws


def compare(base_leg: str, new_leg: str, old_leg: str) -> None:
    """PR-7 read: honest blend(base, new) − honest blend(base, old), same halves."""
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    base, new, old = (_rank(np.load(RESULTS_DIR / f"oof_{n}.npy")) for n in (base_leg, new_leg, old_leg))
    half = np.random.default_rng(0).permutation(len(y)) < len(y) // 2
    s_new, w_new = honest_blend_score(y, base, new, half)
    s_old, w_old = honest_blend_score(y, base, old, half)
    print(f"solo: {new_leg} {roc_auc_score(y, new):.6f}  {old_leg} {roc_auc_score(y, old):.6f}")
    print(f"honest blend with {new_leg}: {s_new:.6f} (w {w_new})")
    print(f"honest blend with {old_leg}: {s_old:.6f} (w {w_old})")
    d = s_new - s_old
    print(f"DELTA {d:+.6f}  bar +{BAR}  → {'ADOPT' if d >= BAR else 'FAIL'}")


def main() -> None:
    if "--base" in sys.argv:
        compare(sys.argv[sys.argv.index("--base") + 1], sys.argv[1], sys.argv[sys.argv.index("--vs") + 1])
        return
    leg = sys.argv[1]
    # Row-order invariant: y comes through build_features.
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    if "--control-blend" in sys.argv:
        control = sum(w * _rank(np.load(RESULTS_DIR / f"oof_{n}.npy")) for n, w in SUBMITTED_BLEND.items())
    else:
        control = np.mean([_rank(np.load(RESULTS_DIR / f"oof_{n}.npy")) for n in CONTROL], axis=0)
    cand = _rank(np.load(RESULTS_DIR / f"oof_{leg}.npy"))
    assert len(cand) == len(y) == len(control)

    print(f"solo {leg}: {roc_auc_score(y, cand):.5f}   control: {roc_auc_score(y, control):.5f}")
    print(f"corr(control, {leg}) = {np.corrcoef(control, cand)[0, 1]:.4f}")

    rng = np.random.default_rng(0)
    half = rng.permutation(len(y)) < len(y) // 2
    deltas = []
    for fit_mask in (half, ~half):
        score_mask = ~fit_mask

        def auc(w: float, mask: np.ndarray) -> float:
            return float(roc_auc_score(y[mask], (1 - w) * control[mask] + w * cand[mask]))

        w = max(GRID, key=lambda g: auc(g, fit_mask))
        d = auc(w, score_mask) - auc(0.0, score_mask)
        deltas.append(d)
        print(f"  fit-half w={w:.2f} → held-out delta {d:+.6f}")
    in_w = max(GRID, key=lambda g: roc_auc_score(y, (1 - g) * control + g * cand))
    in_d = roc_auc_score(y, (1 - in_w) * control + in_w * cand) - roc_auc_score(y, control)
    mean_d = float(np.mean(deltas))
    print(f"in-sample (optimistic): w={in_w:.2f} delta {in_d:+.6f}")
    print(f"HONEST delta {mean_d:+.6f}  bar +{BAR}  → {'ADOPT' if mean_d >= BAR else 'FAIL'}")


if __name__ == "__main__":
    main()
