"""Day-22 pre-registration #5, Phase 3 screen: does the structured-marginalization leg
(`sm_d22`, built by `train_sm.py`) carry anything the deployed inventory does not?

Three reads, in increasing order of what they would cost to act on:

  A  SOLO PROFILE      dw-BA overall / on the 178k missing-driver rows / on complete rows,
                       plus error-overlap with the deployed #1 stack. This is the campaign's
                       standard "is it a core-member clone?" fingerprint.
  B  SEAT TEST         the standard 12-cell split-half LR-stack instrument (honest: decision
                       weights fitted on the fit half, scored on the eval half). Pooled MDE
                       6.5e-5 per `probe_veto_power`. lr8 vs lr8 + sm logits.
  C  M-SLOPE SCREEN    the Phase-3 bar fixed in the pre-registration: mix `sm` into the
                       deployed realmlp breadth at M = 1..8 and read the EXPONENT under the
                       same frozen-w instrument the C1/C2/C3 controls were calibrated on.
                       Exponent inside or below the C2 null band => dead REGARDLESS of its
                       M=1 size. Exponent ~ 0 => it is information, and it is worth its full
                       measured size at the deployed breadth.

Run:  python src/probe_sm_screen.py  ->  results/probe_sm_screen.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SM = "sm_d22"
SEEDS = [42, 7, 123, 2024, 99, 2025, 31, 7777]
LEG = "realmlp_te13_u"
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
N_SPLITS = 6
LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.35]
NULL_LO, NULL_HI = -0.773, -0.060  # C2 null band from probe_mslope.py
OUT = RESULTS_DIR / "probe_sm_screen.txt"


def _decide(y_fit, probs_fit, probs_eval, y_eval) -> float:  # noqa: ANN001
    dw = np.asarray(tune_decision_weights(y_fit, probs_fit))
    return float(balanced_accuracy_score(y_eval, weighted_predict(probs_eval, dw)))


def _arm(X, y, fit_idx, eval_idx):  # noqa: ANN001, ANN202
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(X[fit_idx], y[fit_idx])
    return _decide(y[fit_idx], lr.predict_proba(X[fit_idx]),
                   lr.predict_proba(X[eval_idx]), y[eval_idx])


def _cell(mats, y, f, e):  # noqa: ANN001, ANN202
    return {n: _arm(X, y, f, e) for n, X in mats.items()}


def _blend_cell(base, sm, y, f, e, lams):  # noqa: ANN001, ANN202
    out = {"base": _decide(y[f], base[f], base[e], y[e])}
    for lam in lams:
        mix = (1 - lam) * base + lam * sm
        out[f"lam{lam}"] = _decide(y[f], mix[f], mix[e], y[e])
    return out


def _ba(y, p, w) -> float:  # noqa: ANN001
    return float(balanced_accuracy_score(y, np.argmax(p * w, axis=1)))


def _exponent(ms, d) -> float:  # noqa: ANN001
    if np.any(np.asarray(d) <= 0):
        return float("nan")
    return float(np.polyfit(np.log(ms), np.log(d), 1)[0])


def main() -> None:
    ds = load_dataset()
    y = ds.y
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    sm = np.load(RESULTS_DIR / f"oof_{SM}.npy")
    P = np.stack([np.load(RESULTS_DIR / f"oof_{LEG}_s{s}.npy") for s in SEEDS])
    breadth = P.mean(axis=0)
    u4 = np.load(RESULTS_DIR / "oof_lrstack_u4_r.npy")

    md = np.zeros(len(y), bool)
    for d in ("sleep_duration", "stress_level", "physical_activity_level"):
        md |= ds.train[d].isna().to_numpy()

    say("=" * 78)
    say("probe_sm_screen -- Phase 3: structured marginalization of the missing driver")
    say("=" * 78)

    # ---------------- A. solo profile -------------------------------------------------
    w_sm = robust_decision_weights(y, sm)
    w_u4 = robust_decision_weights(y, u4)
    w_br = robust_decision_weights(y, breadth)
    pred_sm = np.argmax(sm * w_sm, 1)
    pred_u4 = np.argmax(u4 * w_u4, 1)
    say("-- A. solo profile ---------------------------------------------------------")
    say(f"   rows: {len(y)}   missing-driver {md.sum()} ({md.mean():.3f})")
    for nm, p, w in (("sm", sm, w_sm), ("realmlp breadth-8", breadth, w_br), ("u4 stack", u4, w_u4)):
        pr = np.argmax(p * w, 1)
        say(f"   {nm:<18} dw-BA all {balanced_accuracy_score(y, pr):.5f}   "
            f"missing-driver {balanced_accuracy_score(y[md], pr[md]):.5f}   "
            f"complete {balanced_accuracy_score(y[~md], pr[~md]):.5f}")
    e_sm, e_u4 = pred_sm != y, pred_u4 != y
    say(f"   error overlap sm vs u4: {(e_sm & e_u4).sum() / max(e_u4.sum(), 1):.4f} "
        f"of u4's errors are also sm's  |  sm fixes {(~e_sm & e_u4).sum()} of u4's "
        f"{e_u4.sum()} errors ({(~e_sm & e_u4).sum() / max(e_u4.sum(), 1):.3f}), "
        f"{(~e_sm & e_u4 & md).sum() / max((~e_sm & e_u4).sum(), 1):.3f} of those in the "
        f"missing-driver region")
    say()

    # ---------------- B. seat test ----------------------------------------------------
    def logits(key: str) -> NDArray[np.float32]:
        return clipped_logit(np.load(RESULTS_DIR / f"oof_{key}.npy")).astype(np.float32)

    X8 = np.hstack([logits(k) for k in KEYS8])
    mats = {"lr8": X8, "lr_sm": np.hstack([X8, logits(SM)])}
    cells = []
    for seed in range(N_SPLITS):
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                           random_state=seed).split(y, y))
        cells += [(a, b), (b, a)]
    res = Parallel(n_jobs=4)(delayed(_cell)(mats, y, f, e) for f, e in cells)
    s8 = np.array([r["lr8"] for r in res])
    ssm = np.array([r["lr_sm"] for r in res])
    d = ssm - s8
    say("-- B. seat test (12 cells, honest split-half dw; pooled MDE 6.5e-5) ---------")
    say(f"   lr8    {s8.mean():.5f}")
    say(f"   lr_sm  {ssm.mean():.5f}   delta {d.mean():+.5f} "
        f"(sd {d.std():.5f}, positive {int((d > 0).sum())}/{len(d)})")
    say("   per-cell: " + " ".join(f"{x:+.5f}" for x in d))
    say()

    # ---------------- C. blend level (honest) + M-slope (frozen w) --------------------
    say("-- C1. blend level into the realmlp breadth-8 (12 cells, honest dw) ---------")
    resb = Parallel(n_jobs=4)(
        delayed(_blend_cell)(breadth, sm, y, f, e, LAMBDAS) for f, e in cells
    )
    base = np.array([r["base"] for r in resb])
    best_lam, best_delta = None, -1.0
    for lam in LAMBDAS:
        v = np.array([r[f"lam{lam}"] for r in resb])
        dd = v - base
        say(f"   lam={lam:<5} {v.mean():.5f}   delta {dd.mean():+.6f} "
            f"(sd {dd.std():.5f}, positive {int((dd > 0).sum())}/{len(dd)})")
        if dd.mean() > best_delta:
            best_lam, best_delta = lam, float(dd.mean())
    say(f"   base {base.mean():.5f};  best lambda = {best_lam} at {best_delta:+.6f}")
    say()

    say("-- C2. M-SLOPE of the blend gain (frozen w, the instrument the controls "
        "were calibrated on) --")
    lam = best_lam if best_delta > 0 else LAMBDAS[1]
    ms = np.arange(1, 9, dtype=float)
    deltas = []
    for m in range(1, 9):
        stock = P[:m].mean(axis=0)
        w = robust_decision_weights(y, stock)  # frozen per M, shared by both arms
        deltas.append(_ba(y, (1 - lam) * stock + lam * sm, w) - _ba(y, stock, w))
    say(f"   lambda = {lam}")
    say("   M       " + "".join(f"    {int(m)}     " for m in ms))
    say("   delta   " + "".join(f"  {v:+.6f}" for v in deltas))
    if deltas[0] != 0:
        say("   norm.   " + "".join(f"  {v / deltas[0]:+.6f}" for v in deltas))
    e4 = _exponent(ms[:4], deltas[:4])
    e8 = _exponent(ms, deltas)
    say(f"   EXPONENT M=1..4 {e4:+.3f}   M=1..8 {e8:+.3f}   "
        f"(C2 null band [{NULL_LO:+.3f}, {NULL_HI:+.3f}])")
    if np.isnan(e4):
        v = "UNDEFINED (delta crosses zero) -- no gain to characterise"
    elif e4 > NULL_HI:
        v = "MEAN SHIFT -- survives averaging; read the LEVEL against the deliverable bar"
    elif e4 < NULL_LO:
        v = "VARIANCE REDUCTION -- worth ~1/M, dead at the deployed breadth"
    else:
        v = "INSIDE THE NULL BAND -- indistinguishable from selection noise"
    say(f"   VERDICT: {v}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
