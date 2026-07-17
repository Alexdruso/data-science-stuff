"""Day-14 — red-team fix F1: the hypothesis-class cells rerun WITHOUT the
in-sample decision-weight confound.

The Day-13 coverage-matrix vetoes of the high-capacity metas were measured
with decision weights tuned on IN-SAMPLE fit-half probabilities
(fit_gbdtmeta/fit_nnmeta return clf.predict_proba(Xtr) on their own training
rows; the band arm's second LR likewise) — overfit probs => miscalibrated dw
=> a handicap the 24-param LR incumbent doesn't suffer. Those were the only
coverage-matrix cells measured with a handicapped challenger; the "LR is a
local optimum across hypothesis classes" plank rests on them.

Honest protocol (this probe): within half A, 3-fold CROSS-FIT the meta and
tune dw on the cross-fitted probs; eval probs come from a full-half-A fit.
Same treatment for every arm (LR included, for symmetry). Same 12 split-half
cells as always.

Arms: lr (in-sample dw, deployed treatment — continuity anchor), lr_h,
gbdt_h, nn_h, band_h. Challenger deltas read vs lr_h. Confounded reference
reads (Day-13): gbdt -0.00009 (2/12), nn -0.00015 (1/12), band -0.00014
(2/12). A flip to >= +0.0003 at >=10/12 fells the hypothesis-class plank.

Run: python src/probe_meta_honest.py -> results/probe_meta_honest.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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
TOP2 = ["hgbc_te3_u_breadth", "realmlp_te13_u_breadth"]


def _score(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _fit_lr_probs(blocks, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    X = np.hstack(blocks)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])


def _crossfit(fn, blocks, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    """Cross-fitted probs on half A (for dw tuning) + full-half-A eval probs."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cf = np.zeros((len(fit_idx), 3))
    for sub, hold in skf.split(fit_idx, y[fit_idx]):
        _, ph = fn(blocks, y, fit_idx[sub], fit_idx[hold])
        cf[hold] = ph
    _, pe = fn(blocks, y, fit_idx, eval_idx)
    return cf, pe


def _fit_band(blocks, y, fit_idx, eval_idx, top2_am):  # noqa: ANN001, ANN202
    """Two-regime band meta with the second-stage LR cross-fit inside half A."""
    X = np.hstack(blocks)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    pf, pe = lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])

    def band_mask(probs, idx):  # noqa: ANN001, ANN202
        srt = np.sort(probs, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        return (top2_am[idx, 0] != top2_am[idx, 1]) | (margin < 0.1)

    bf, be = band_mask(pf, fit_idx), band_mask(pe, eval_idx)
    pf2, pe2 = pf.copy(), pe.copy()
    # cross-fit the band LR within half A (the confound in the Day-13 read)
    band_abs = fit_idx[bf]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    band_pos = np.where(bf)[0]
    for sub, hold in skf.split(band_abs, y[band_abs]):
        blr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        blr.fit(X[band_abs[sub]], y[band_abs[sub]])
        pf2[band_pos[hold]] = blr.predict_proba(X[band_abs[hold]])
    blr_full = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    blr_full.fit(X[band_abs], y[band_abs])
    pe2[be] = blr_full.predict_proba(X[eval_idx][be])
    return pf2, pe2


def cell(blocks, top2_am, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    out = {}
    # deployed-treatment LR (continuity anchor: dw on in-sample probs)
    pf, pe = _fit_lr_probs(blocks, y, fit_idx, eval_idx)
    out["lr"] = _score(y[fit_idx], pf, pe, y[eval_idx])
    # honest arms: dw on 3-fold cross-fitted half-A probs
    cf, pe_h = _crossfit(_fit_lr_probs, blocks, y, fit_idx, eval_idx)
    out["lr_h"] = _score(y[fit_idx], cf, pe_h, y[eval_idx])
    for name, fn in [("gbdt_h", fit_gbdtmeta), ("nn_h", fit_nnmeta)]:
        cf, pe_m = _crossfit(fn, blocks, y, fit_idx, eval_idx)
        out[name] = _score(y[fit_idx], cf, pe_m, y[eval_idx])
    cf_b, pe_b = _fit_band(blocks, y, fit_idx, eval_idx, top2_am)
    out["band_h"] = _score(y[fit_idx], cf_b, pe_b, y[eval_idx])
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = [
        clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    ]
    top2_am = np.stack(
        [np.load(RESULTS_DIR / f"oof_{k}.npy").argmax(axis=1) for k in TOP2], axis=1
    )

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=2)(
        delayed(cell)(blocks, top2_am, y, f, e) for f, e in cells
    )
    arm_names = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arm_names}
    lines = [
        f"honest-meta rerun (red-team F1), {len(cells)} cells; legs={KEYS8}",
        "  dw tuned on 3-fold cross-fitted half-A probs for every *_h arm "
        "(Day-13 confounded reads: gbdt -0.00009, nn -0.00015, band -0.00014)",
    ]
    for a in sorted(arm_names, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr_h"]
        lines.append(
            f"  {a:<7} {scores[a].mean():.5f}   vs lr_h {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    lines.append(
        "  cells (deltas vs lr_h): "
        + "; ".join(
            f"{a}=[{', '.join(f'{v:+.5f}' for v in scores[a] - scores['lr_h'])}]"
            for a in arm_names
            if a != "lr_h"
        )
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_meta_honest.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
