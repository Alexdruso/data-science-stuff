"""Fast per-class decision-weight fitter + balanced accuracy, for the Day-22 M-slope probes.

WHY THIS EXISTS. `kaggle_utils.tune_decision_weights` runs Nelder-Mead with
`maxiter=3000, xatol=fatol=1e-7` on `balanced_accuracy(argmax(p * w))` — a **piecewise
constant** objective. NM cannot reach a 1e-7 tolerance on a step function, so every call
burns close to the full budget (15 restarts x up to 3000 iterations ~ 45k objective
evaluations). That is fine for the handful of calls a normal probe makes; the M-slope
probes need hundreds, and it turned a 5-minute read into a multi-hour one.

Two changes, neither of which alters what is being estimated:

  * `ba_fast`   — balanced accuracy via two `bincount`s instead of
                  `sklearn.balanced_accuracy_score` (which builds a confusion matrix).
  * `fit_dw`    — coarse-to-fine GRID search over the 2 free log-weights (class 0 pinned
                  at 1; the problem is scale-invariant). A grid is the right tool for a
                  piecewise-constant surface: it cannot be fooled by flat regions the way
                  a simplex method is, and the campaign already measured that this
                  surface is *nearly flat* in w (Day-20: w vectors 0.2 apart score within
                  2e-5), so grid resolution is not the binding constraint.

Both arms of every paired read use the SAME fitter, so any residual sub-optimality is
common-mode and cancels in the delta — which is all the M-slope probes consume.

`selftest()` checks `fit_dw` against the deployed NM-fitted weights on a real leg.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ba_fast(y: NDArray[np.int64], pred: NDArray[np.int64], counts: NDArray[np.int64]) -> float:
    """Balanced accuracy = mean per-class recall. `counts` = np.bincount(y)."""
    hits = np.bincount(y[pred == y], minlength=len(counts))
    return float(np.mean(hits / counts))


def score_w(
    y: NDArray[np.int64],
    p: NDArray[np.float64],
    w: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> float:
    return ba_fast(y, np.argmax(p * w, axis=1), counts)


def fit_dw(
    y: NDArray[np.int64],
    p: NDArray[np.float64],
    grid: int = 17,
    rounds: int = 3,
    span: float = 2.4,
    fit_n: int = 0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Coarse-to-fine grid over log-weights for classes 1..K-1 (class 0 pinned at 1).

    `fit_n` subsamples the rows used to FIT w (the weight vector has only K-1 free
    parameters, so 120k rows is far past the point where more rows change the optimum).
    Scoring is always done by the caller on the full eval split. Applied identically to
    both arms of a paired read, so it is common-mode.
    """
    k = p.shape[1]
    assert k == 3, "the log-ratio fast path assumes 3 classes"
    if fit_n and len(y) > fit_n:
        idx = np.random.default_rng(seed).choice(len(y), fit_n, replace=False)
        y, p = y[idx], p[idx]
    counts = np.bincount(y, minlength=k).astype(np.float64)
    # Decision depends only on the two log-ratios: with weights (1, a, b), class 1 wins
    # iff u+la > 0 and u+la >= v+lb; class 2 iff v+lb > 0 and v+lb > u+la; else class 0.
    # Precomputing u, v makes each grid evaluation two adds + compares instead of a
    # 3-column multiply + argmax.
    lp = np.log(np.clip(p, 1e-12, None)).astype(np.float32)
    u = lp[:, 1] - lp[:, 0]
    v = lp[:, 2] - lp[:, 0]
    y32 = y.astype(np.int8)
    m0, m1, m2 = y32 == 0, y32 == 1, y32 == 2

    def ba_at(la: float, lb: float) -> float:
        a, b = u + np.float32(la), v + np.float32(lb)
        one = (a > 0) & (a >= b)
        two = (b > 0) & (b > a)
        zero = ~(one | two)
        return float(
            (np.count_nonzero(zero & m0) / counts[0]
             + np.count_nonzero(one & m1) / counts[1]
             + np.count_nonzero(two & m2) / counts[2]) / 3.0
        )

    center = np.zeros(2)
    for _ in range(rounds):
        ax0 = np.linspace(center[0] - span / 2, center[0] + span / 2, grid)
        ax1 = np.linspace(center[1] - span / 2, center[1] + span / 2, grid)
        best, best_pt = -1.0, center
        for la in ax0:
            for lb in ax1:
                s = ba_at(la, lb)
                if s > best:
                    best, best_pt = s, np.array([la, lb])
        center, span = best_pt, span / (grid / 3.0)
    best_w = np.exp(np.concatenate([[0.0], center]))
    return np.asarray(best_w / best_w.max(), dtype=np.float64)


def selftest() -> None:  # pragma: no cover - diagnostic entry point
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from sklearn.metrics import balanced_accuracy_score

    from train_common import RESULTS_DIR, load_dataset

    y = load_dataset().y
    counts = np.bincount(y, minlength=3)
    P = np.stack([
        np.load(RESULTS_DIR / f"oof_realmlp_te13_u_s{s}.npy")
        for s in (42, 7, 123, 2024, 99, 2025, 31, 7777)
    ])
    breadth = P.mean(0)
    nm_w = np.array([1.0, 0.94941, 0.8001])  # robust_decision_weights, cached Day-22
    nm_s = balanced_accuracy_score(y, np.argmax(breadth * nm_w, 1))
    import time

    t0 = time.time()
    g_w = fit_dw(y, breadth)
    dt = time.time() - t0
    g_s = balanced_accuracy_score(y, np.argmax(breadth * g_w, 1))
    print(f"NM   w={np.round(nm_w, 5).tolist()}  BA={nm_s:.6f}")
    print(f"grid w={np.round(g_w, 5).tolist()}  BA={g_s:.6f}  ({dt:.1f}s)")
    print(f"delta grid - NM = {g_s - nm_s:+.6f}   (>=0 means the grid found at least as good a w)")
    assert abs(ba_fast(y, np.argmax(breadth * g_w, 1), counts) - g_s) < 1e-12, "ba_fast mismatch"
    print("ba_fast matches sklearn exactly.")


if __name__ == "__main__":
    selftest()
