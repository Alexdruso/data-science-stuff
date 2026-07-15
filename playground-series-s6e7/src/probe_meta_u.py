"""Day-13 PM — u-surface meta tournament: the last unmeasured combiner cell.

Day-8's tournament (tree/NN metas lose to LR by ~0.0002) ran on the OLD
repaired-surface leg inventory; the metablend verdict later FLIPPED SIGN on the
u surface, so the meta-family ranking was never re-established on the current
inventory (u legs, 8-seed breadths). Also LR's C=1.0 has never been swept
anywhere. Same 12 split-half cells as every seat test (meta + decision weights
refit on half A, scored on half B), arms:

  lr     — LR-on-logits C=1.0 (incumbent)
  lrC03  — C=0.3 (more shrinkage)
  lrC3   — C=3.0 (less shrinkage)
  gbdt   — ridgecal->LGBM meta (build_alt_meta.fit_gbdtmeta)
  nn     — oversampled MLP meta (build_alt_meta.fit_nnmeta)

Adoption bar: >= +0.0003 mean over lr at >=10/12 positive (the lr6 precedent).
Prior: veto (3x no-regional-trust results), but this cell is unmeasured.

Run: python src/probe_meta_u.py -> results/probe_meta_u.txt
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
from build_alt_meta import fit_gbdtmeta, fit_nnmeta
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


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _lr_arm(X, y, fit_idx, eval_idx, c):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return _decide(
        y[fit_idx], lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx]),
        y[eval_idx],
    )


def cell(X, blocks, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    out = {}
    for name, c in [("lr", 1.0), ("lrC03", 0.3), ("lrC3", 3.0)]:
        out[name] = _lr_arm(X, y, fit_idx, eval_idx, c)
    for name, fn in [("gbdt", fit_gbdtmeta), ("nn", fit_nnmeta)]:
        pf, pe = fn(blocks, y, fit_idx, eval_idx)
        out[name] = _decide(y[fit_idx], pf, pe, y[eval_idx])
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = [
        clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    ]
    X = np.hstack(blocks)

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=2)(
        delayed(cell)(X, blocks, y, f, e) for f, e in cells
    )
    arm_names = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arm_names}
    lines = [f"u-surface meta tournament, {len(cells)} cells; legs={KEYS8}"]
    for a in sorted(arm_names, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr"]
        lines.append(
            f"  {a:<6} {scores[a].mean():.5f}   vs lr {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_meta_u.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
