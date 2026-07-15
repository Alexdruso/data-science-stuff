"""Day-13 PM — combiner STRUCTURE arms (last cells of the coverage matrix).

After today's tournament (hypothesis class: gbdt/nn metas lose; C sweep inert)
and polish probe (fitting objective +0.00006; calibration shape +0.00004),
two structural reshapes remain unmeasured:

  pair  — pairwise-coupled binaries: three one-vs-one LRs over the same 24
          logits, Price-coupled to a 3-class posterior. Decouples the
          at-risk<->unh boundary from softmax normalization sharing.
          (Distinct from the cascade kill = base-training level, and from the
          specialist-binaries kill = extra COLUMNS into one multinomial meta.)
  band  — two-regime meta conditioned on PREDICTION GEOMETRY (not x): a second
          LR fit only on rows where the top-2 legs' argmaxes disagree or the
          base-LR margin < 0.1; base LR elsewhere. (All three regional-trust
          vetoes were x-conditioned.)

Same 12 split-half cells; bar >= +0.0003 at >=10/12.

Run: python src/probe_struct_u.py -> results/probe_struct_u.txt
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
TOP2 = ["hgbc_te3_u_breadth", "realmlp_te13_u_breadth"]


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def price_couple(r: dict) -> np.ndarray:
    """Price's coupling: P_c ∝ 1 / (sum_{j!=c} 1/r_cj - (K-2)); r_cj = P(c | c or j)."""
    n = len(next(iter(r.values())))
    p = np.zeros((n, 3))
    for c in range(3):
        inv = np.zeros(n)
        for j in range(3):
            if j == c:
                continue
            r_cj = r[(c, j)] if (c, j) in r else 1.0 - r[(j, c)]
            inv += 1.0 / np.clip(r_cj, 1e-6, 1 - 1e-6)
        p[:, c] = 1.0 / np.clip(inv - 1.0, 1e-6, None)
    return p / p.sum(axis=1, keepdims=True)


def cell(X, top2_am, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    out = {}
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    pf, pe = lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx])
    out["lr"] = _decide(y[fit_idx], pf, pe, y[eval_idx])

    # pairwise-coupled binaries
    rf, re = {}, {}
    for a in range(3):
        for b in range(a + 1, 3):
            m = np.isin(y[fit_idx], [a, b])
            blr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            blr.fit(X[fit_idx][m], (y[fit_idx][m] == a).astype(int))
            rf[(a, b)] = blr.predict_proba(X[fit_idx])[:, 1]
            re[(a, b)] = blr.predict_proba(X[eval_idx])[:, 1]
    out["pair"] = _decide(y[fit_idx], price_couple(rf), price_couple(re), y[eval_idx])

    # disagreement-band two-regime meta (prediction-geometry conditioned)
    def band_mask(probs, idx):  # noqa: ANN001, ANN202
        margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
        return (top2_am[idx, 0] != top2_am[idx, 1]) | (margin < 0.1)

    bf, be = band_mask(pf, fit_idx), band_mask(pe, eval_idx)
    blr2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    blr2.fit(X[fit_idx][bf], y[fit_idx][bf])
    pf2, pe2 = pf.copy(), pe.copy()
    pf2[bf] = blr2.predict_proba(X[fit_idx][bf])
    pe2[be] = blr2.predict_proba(X[eval_idx][be])
    out["band"] = _decide(y[fit_idx], pf2, pe2, y[eval_idx])
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    }
    X = np.hstack([blocks[k] for k in KEYS8])
    top2_am = np.stack(
        [np.load(RESULTS_DIR / f"oof_{k}.npy").argmax(axis=1) for k in TOP2], axis=1
    )

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(X, top2_am, y, f, e) for f, e in cells
    )
    arm_names = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arm_names}
    lines = [f"combiner structure arms, {len(cells)} cells"]
    for a in sorted(arm_names, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr"]
        lines.append(
            f"  {a:<5} {scores[a].mean():.5f}   vs lr {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_struct_u.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
