"""Cross-seed dispersion seat test (Day-16 brainstorm P2).

Every combiner the campaign measured consumes breadth-AVERAGED logits — the
averaging destroys within-family seed dispersion before any meta sees it.
Dispersion = per-row per-class std across the 8 canonical seeds of each
te-family (hgbc_te3 / lgbm_te3 / realmlp_te13) = 9 columns, z-scored. The
claim: seed disagreement marks epistemic-boundary rows, letting the LR
discount the mean where members disagree (a trust channel keyed on model
uncertainty, unlike the flat missingness-keyed regime metas).

Standard 12-cell split-half instrument (probe_seat_bins pattern):
  lr8  — incumbent lrstack leg set
  lrd  — lr8 + 9 dispersion columns
Pooled-veto MDE 6.5e-5 (Day-14): a null is a clean closure.

Run: python src/probe_seat_disp.py -> results/probe_seat_disp.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

N_SPLITS = 6
KEYS8 = [
    "lgbm_bagw",
    "hgbc_bagw",
    "xgboost_bagw",
    "catboost_bagw",
    "mlp",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
    "realmlp_te13_u_breadth",
]
SEEDS8 = [42, 7, 123, 2024, 99, 2025, 31, 7777]
FAMILIES = ["hgbc_te3", "lgbm_te3", "realmlp_te13"]


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _arm(X, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return _decide(
        y[fit_idx], lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx]),
        y[eval_idx],
    )


def cell(X8, Xd, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {
        "lr8": _arm(X8, y, fit_idx, eval_idx),
        "lrd": _arm(Xd, y, fit_idx, eval_idx),
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    X8 = np.hstack(
        [
            clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
            for k in KEYS8
        ]
    )
    disp_blocks = []
    for fam in FAMILIES:
        stack = np.stack(
            [np.load(RESULTS_DIR / f"oof_{fam}_u_s{s}.npy") for s in SEEDS8], axis=0
        )
        disp_blocks.append(stack.std(axis=0).astype(np.float32))
    disp = np.hstack(disp_blocks)
    disp = (disp - disp.mean(axis=0)) / (disp.std(axis=0) + 1e-9)
    Xd = np.hstack([X8, disp])
    print(f"lr8 {X8.shape} + dispersion {disp.shape}", flush=True)

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(delayed(cell)(X8, Xd, y, f, e) for f, e in cells)
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [
        f"dispersion seat test, {len(cells)} cells; lr8={KEYS8}",
        f"+9 z-scored per-class seed-std cols from {FAMILIES} x seeds {SEEDS8}",
    ]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    lines.append("per-cell deltas (Day-14 hygiene rule):")
    d = scores["lrd"] - scores["lr8"]
    lines.append("  " + " ".join(f"{x:+.5f}" for x in d))
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_seat_disp.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
