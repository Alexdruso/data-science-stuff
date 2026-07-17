"""Day-14 — red-team fixes F2+F3: adv-weighted seat instrument, breadth-matched
FTT, and the first STACKED micro-arm.

Three measured soft spots in the seat-test family (Day-14 red-team audit):
  (F3) challengers entered single-seed vs 8-seed-breadth incumbents — the ftt
       veto (lr9 -0.00001, Day-10) used s42 alone while s42/7/123/2024 exist;
  (F2a) all cells score PLAIN bacc on a train holdout — a challenger whose
       effect concentrates on test-like rows is diluted ~3.3x (test-like =
       top-30% adv score; adv-AUC 0.654). Second scoring pass added: per-class
       recall importance-weighted by adv-odds w = s/(1-s) on the eval half;
  (F2b) the persistent micro-positive vetoed arms (lrrt1 +0.00005 9/12,
       basin lrt +0.00003 8/12, alpha_c +0.00004 8/12) were never STACKED.

Arms (same 12 split-half cells as every seat probe; LR meta + decision weights
fit on half A, both scores read on half B):
  lr8     — incumbent leg set (lrstack_u4 shape)
  lrftt4  — lr8 + mean(ftt_te13_u s42/s7/s123/s2024)   [F3]
  lrrt1   — lr8 + hgbc_te3_rt_u_s42                     [replication]
  lrmicro — lr8 + rt1 + hgbc_te3_u_s42t legs, alpha_c per-class power
            calibration on top of the meta probs        [F2b stack]

Adoption bar unchanged: >= +0.0003 at >= 10/12 on the PLAIN read. The advw
read is diagnostic: micro-positives that vanish under adv weighting are
MCAR-parked (not transferable); ones that grow are being diluted.

Run: python src/probe_seat_day14.py -> results/probe_seat_day14.txt
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
FTT_SEEDS = ["ftt_te13_u_s42", "ftt_te13_u_s7", "ftt_te13_u_s123", "ftt_te13_u_s2024"]
RT1 = "hgbc_te3_rt_u_s42"
BASIN = "hgbc_te3_u_s42t"


def _advw_bacc(y_eval, pred, w_eval) -> float:  # noqa: ANN001
    """Per-class recall, importance-weighted by adversarial odds."""
    recalls = []
    for c in np.unique(y_eval):
        m = y_eval == c
        recalls.append(float(np.average(pred[m] == c, weights=w_eval[m])))
    return float(np.mean(recalls))


def _decide(y_fit, probs_fit, probs_eval, y_eval, w_eval):  # noqa: ANN001, ANN202
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    pred = weighted_predict(probs_eval, dw)
    return (
        float(balanced_accuracy_score(y_eval, pred)),
        _advw_bacc(y_eval, pred, w_eval),
    )


def _fit_lr(X, y, fit_idx):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return lr


def _alpha_calibrate(pf, pe, y_fit):  # noqa: ANN001, ANN202
    """Per-class power calibration argmax(w_c * p_c^a_c), fit on half A."""
    lpf = np.log(np.clip(pf, 1e-9, 1))
    lpe = np.log(np.clip(pe, 1e-9, 1))

    def obj(theta: np.ndarray) -> float:
        a, lw = theta[0:3], theta[3:6]
        return -float(balanced_accuracy_score(y_fit, (lpf * a + lw).argmax(axis=1)))

    res = minimize(
        obj, np.concatenate([np.ones(3), np.zeros(3)]),
        method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-6},
    )
    a, lw = res.x[0:3], res.x[3:6]
    return lpe * a + lw


def cell(arms, micro_X, y, adv_w, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    out = {}
    w_eval = adv_w[eval_idx]
    for name, X in arms.items():
        lr = _fit_lr(X, y, fit_idx)
        out[name] = _decide(
            y[fit_idx], lr.predict_proba(X[fit_idx]), lr.predict_proba(X[eval_idx]),
            y[eval_idx], w_eval,
        )
    # stacked micro-arm: widest leg set + alpha_c on top (argmax read; the
    # power calibration subsumes the per-class decision weights)
    lr = _fit_lr(micro_X, y, fit_idx)
    scores_e = _alpha_calibrate(
        lr.predict_proba(micro_X[fit_idx]), lr.predict_proba(micro_X[eval_idx]),
        y[fit_idx],
    )
    pred = scores_e.argmax(axis=1)
    out["lrmicro"] = (
        float(balanced_accuracy_score(y[eval_idx], pred)),
        _advw_bacc(y[eval_idx], pred, w_eval),
    )
    return out


def main() -> None:
    ds = load_dataset()
    y = ds.y
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    assert adv.shape[0] == len(y)
    adv_w = np.clip(adv, 1e-3, 1 - 1e-3)
    adv_w = adv_w / (1.0 - adv_w)

    blocks = {
        k: clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")).astype(np.float32)
        for k in [*KEYS8, RT1, BASIN]
    }
    ftt4 = clipped_logit(
        np.mean([np.load(RESULTS_DIR / f"oof_{k}.npy") for k in FTT_SEEDS], axis=0)
    ).astype(np.float32)
    X8 = np.hstack([blocks[k] for k in KEYS8])
    arms = {
        "lr8": X8,
        "lrftt4": np.hstack([X8, ftt4]),
        "lrrt1": np.hstack([X8, blocks[RT1]]),
    }
    micro_X = np.hstack([X8, blocks[RT1], blocks[BASIN]])

    cells = []
    for seed in range(N_SPLITS):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y, y))
        cells.append((a, b))
        cells.append((b, a))

    results = Parallel(n_jobs=4)(
        delayed(cell)(arms, micro_X, y, adv_w, f, e) for f, e in cells
    )
    arm_names = list(results[0].keys())
    plain = {a: np.array([r[a][0] for r in results]) for a in arm_names}
    advw = {a: np.array([r[a][1] for r in results]) for a in arm_names}
    lines = [
        f"day-14 seat probe (red-team F2+F3), {len(cells)} cells; lr8={KEYS8}",
        f"  lrftt4=+mean({len(FTT_SEEDS)} ftt seeds)  lrrt1=+{RT1}  "
        f"lrmicro=+{RT1}+{BASIN}+alpha_c",
        "  plain = deployed read (adoption bar +0.0003 at >=10/12); "
        "advw = adv-odds-weighted per-class recall (diagnostic)",
    ]
    for a in sorted(arm_names, key=lambda a: -plain[a].mean()):
        dp = plain[a] - plain["lr8"]
        da = advw[a] - advw["lr8"]
        lines.append(
            f"  {a:<8} plain {plain[a].mean():.5f} vs lr8 {dp.mean():+.5f} "
            f"(sd {dp.std():.5f}, positive {int((dp > 0).sum())}/12)  |  "
            f"advw {advw[a].mean():.5f} d {da.mean():+.5f} "
            f"(sd {da.std():.5f}, positive {int((da > 0).sum())}/12)"
        )
    lines.append(
        "  cells (plain deltas): "
        + "; ".join(
            f"{a}=[{', '.join(f'{v:+.5f}' for v in plain[a] - plain['lr8'])}]"
            for a in arm_names
            if a != "lr8"
        )
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_seat_day14.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
