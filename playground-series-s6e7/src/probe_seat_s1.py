"""S1 seat test (Day-11) — second half of the pre-registered read.

Even though the error-placement leg is degraded solo (the penalty sacrifices
accuracy on complete∩test-like rows), its error-overlap with the core collapsed
(~92% → ~63%): it is the most decorrelated NN of the campaign. The mlp_la_r
precedent says a per-class LR meta can extract a seat from a decorrelated-but-weak
leg where scalar NM cannot. 12 split-half cells, same protocol as
probe_combiner_ftt / probe_seat_bins:

  lr8   — the lrstack_u3 leg set (incumbent shape)
  lrs1  — lr8 + the S1 leg (3-class logits)

Run: python src/probe_seat_s1.py <s1_key> [<s1_key> ...]
  -> results/probe_seat_s1.txt
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
    "lgbm_bagw", "hgbc_bagw", "xgboost_bagw", "catboost_bagw", "mlp",
    "hgbc_te3_u_breadth", "lgbm_te3_u_breadth", "realmlp_te13_u_breadth",
]


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


def main() -> None:
    keys = sys.argv[1:]
    if not keys:
        raise SystemExit("usage: probe_seat_s1.py <s1_key> ...")
    ds = load_dataset()
    y = ds.y
    base = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in KEYS8
    }
    X8 = np.hstack([base[k] for k in KEYS8])
    s1_blocks = {
        k: np.hstack([X8, clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)])
        for k in keys
    }

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells += [(a, b), (b, a)]

    def cell(fit_idx, eval_idx):  # noqa: ANN001, ANN202
        out = {"lr8": _arm(X8, y, fit_idx, eval_idx)}
        for k in keys:
            out[k] = _arm(s1_blocks[k], y, fit_idx, eval_idx)
        return out

    results = Parallel(n_jobs=4)(delayed(cell)(f, e) for f, e in cells)
    arms = list(results[0].keys())
    scores = {a: np.array([r[a] for r in results]) for a in arms}
    lines = [f"S1 seat test, {len(cells)} cells; lr8 + each S1 leg"]
    for a in sorted(arms, key=lambda a: -scores[a].mean()):
        d = scores[a] - scores["lr8"]
        lines.append(
            f"  {a:<20} {scores[a].mean():.5f}   vs lr8 {d.mean():+.5f} "
            f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})"
        )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_seat_s1.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
