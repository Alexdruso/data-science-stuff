"""Config-basin seat test (Day-12 PM): does a DIFFERENT-BASIN hgbc_te3 seat?

Day-12's Optuna found a second config basin (77 leaves / lr 0.034 / mf 0.41 /
bins 192 vs Susol's 33 / 0.063 / 0.82 / 237) — flat solo (0.9505 vs 0.9504,
hgbc_te3_u_s42t) but differently regularized, so its ERRORS may sit elsewhere.
All seven prior LR-seat vetoes tested architecture/feature/objective diversity;
config-basin diversity within one architecture was never read. Same 12
split-half cells as probe_seat_bins.py:

  lr8   — the lrstack_u3 leg set (incumbent shape)
  lrt   — lr8 + the basin leg's 3 logit columns (hgbc_te3_u_s42t)

Expected flat per the 3-dim label-channel model (same vehicle class, same
information channel) — but it's the day's natural closing read.

Run: python src/probe_seat_basin.py -> results/probe_seat_basin.txt
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
BASIN = "hgbc_te3_u_s42t"


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


def cell(X8, Xt, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    return {
        "lr8": _arm(X8, y, fit_idx, eval_idx),
        "lrt": _arm(Xt, y, fit_idx, eval_idx),
    }


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    }
    basin = clipped_logit(np.load(RESULTS_DIR / f"oof_{BASIN}.npy")).astype(np.float32)
    X8 = np.hstack([blocks[k] for k in KEYS8])
    Xt = np.hstack([X8, basin])

    # decorrelation fingerprint vs the incumbent hgbc_te3 breadth leg
    inc = np.load(RESULTS_DIR / "oof_hgbc_te3_u_breadth.npy")
    bas = np.load(RESULTS_DIR / f"oof_{BASIN}.npy")
    inc_err = inc.argmax(1) != y
    bas_err = bas.argmax(1) != y
    overlap = (inc_err & bas_err).sum() / max(1, (inc_err | bas_err).sum())
    fixes = int((inc_err & ~bas_err).sum())
    breaks = int((~inc_err & bas_err).sum())
    fingerprint = (
        f"vs hgbc_te3_u_breadth (argmax): err-overlap {overlap:.3f}, "
        f"fixes {fixes} / breaks {breaks}"
    )
    print(fingerprint)

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X8, Xt, y, f, e) for f, e in cells
    )
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [
        f"config-basin seat test, {len(cells)} cells; lr8={KEYS8}, +{BASIN}",
        fingerprint,
    ]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<4} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_seat_basin.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
