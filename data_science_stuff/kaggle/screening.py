"""Mechanism screening for ensemble levers — the M-slope (decay-exponent) instrument.

**The problem this solves.** You have a deployed ensemble that averages ``M``
members (seeds, folds, model variants) and a candidate lever — a new feature
block, a loss tweak, a config change. You measure it on one member, it wins at
every seed with a tiny standard error, and you conclude it is real. That
inference is invalid. A *variance reducer* lowers each member's posterior
noise, so it is significantly positive at every individual member **by
construction**, and it is worth nothing once you average ``M`` members, because
averaging removes the same noise from the incumbent arm too. Per-seed
significance cannot distinguish "adds information" from "removes noise I was
about to remove anyway" (s6e7 Day-22: a lever at 11 standard errors, 4/4 seeds
positive, was worth +0.00008 at M=4 and nothing at the deployed M=8).

**The instrument.** Measure the paired delta at ``M = 1, 2, 3, 4`` and fit the
exponent ``b`` of ``|delta| ~ M**b``:

===================================  ==============================  ========
mechanism                            why                             ``b``
===================================  ==============================  ========
information                          survives averaging (linearity)  ``> 0``
selection noise (winner's curse)     difference of means, CLT        ``-0.5``
variance reduction                   competes with averaging         ``-1``
===================================  ==============================  ========

**Averaging reveals information and destroys noise.** A real lever's advantage
does not merely survive averaging, it *grows*: at ``M = 1`` both arms are noisy
and member noise randomly throws boundary rows to the wrong side, masking a
systematic edge. s6e7's single largest genuine find read ``b = +0.558``; the
11-sigma variance reducer read ``-1.107``; the synthetic controls landed on
``-0.027`` / ``-0.417`` / ``-1.002`` against point predictions of 0 / -0.5 / -1.

Screen by the exponent first and the magnitude second. Four members and minutes
of numpy against arrays already on disk replace rebuilding a candidate at full
deployed size.

**Calibrate before you trust the scale.** The verdict boundaries here are
nearest-anchor cuts between the two *robust* anchors (``0`` and ``-1``, which
separated by 11 standard deviations under three different decision layers).
The selection-noise band is **instrument-dependent** — s6e7 measured it at
-0.33, -0.85 and -1.54 under frozen-weight, per-arm-weight and plain-argmax
scoring of the same control — so this module ships no hard-coded null band.
Derive one on your own data with :func:`selection_null_arms`, and pass it to
:func:`classify_exponent`. :func:`mean_shift_control` and
:func:`variance_reduction_control` reproduce the known-real and known-fake
anchors; magnitude-match them with :func:`calibrate_lambda` before comparing
exponents, since an exponent difference between a large and a small lever is
confounded with size.

Full derivation and the measured numbers:
``playground-series-s6e7/notes/mslope_writeup.md``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedShuffleSplit

from data_science_stuff.kaggle.decision import (
    balanced_accuracy_fast,
    tune_decision_weights_grid,
)

#: Exponent of a lever that adds information. The oracle case (a uniform shift
#: toward the truth, already fully expressed at one member) sits at exactly 0;
#: a realistic information lever, partly masked by member noise at M=1, reads
#: above it.
INFORMATION_ANCHOR = 0.0
#: Exponent of a sign picked by the M=1 read — a difference of two means.
SELECTION_NOISE_ANCHOR = -0.5
#: Exponent of a pure variance reducer: it competes directly with averaging.
VARIANCE_REDUCTION_ANCHOR = -1.0

# Two points define a line; fewer has no slope. (Four or more is the advice —
# see :func:`decay_exponent`.)
MIN_SIZES_FOR_SLOPE = 2

INFORMATION = "information — worth its M=1 size or more at the deployed breadth"
SLOW_DECAY = (
    "slow decay — not information (it does not grow under averaging) but far "
    "slower than a variance reducer; report, do not adopt"
)
SELECTION_NOISE = "indistinguishable from selection noise — dead"
NOT_INFORMATION = (
    "not information — consistent with variance reduction; dead at the deployed breadth"
)
UNDEFINED = "undefined (the delta crosses zero) — not evidence of information"

#: Scores one paired arm: ``(y_fit, proba_fit, y_eval, proba_eval) -> score``.
#: The fit split exists so the decision layer can be re-fitted honestly per arm.
DecisionLayer = Callable[
    [
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.float64],
    ],
    float,
]


def honest_weighted_layer(
    y_fit: NDArray[np.int64],
    proba_fit: NDArray[np.float64],
    y_eval: NDArray[np.int64],
    proba_eval: NDArray[np.float64],
) -> float:
    """Per-arm decision weights fitted on the fit half, scored on the eval half.

    **The default, and the only admissible layer when the two arms are
    different feature matrices.** Freezing one arm's weights across both is
    neutral only when the challenger is a perturbation of the stock arm (both
    share a calibration); two different matrices have differently calibrated
    posteriors, and s6e7 watched that confound flip a lever's sign from
    +0.000206 to -0.000027. Each arm gets its own weights — as at deployment —
    with no in-sample optimism.
    """
    weights = tune_decision_weights_grid(y_fit, proba_fit)
    pred = np.asarray(np.argmax(proba_eval * weights, axis=1), dtype=np.int64)
    return balanced_accuracy_fast(y_eval, pred, n_classes=proba_eval.shape[1])


def argmax_layer(
    _y_fit: NDArray[np.int64],
    _proba_fit: NDArray[np.float64],
    y_eval: NDArray[np.int64],
    proba_eval: NDArray[np.float64],
) -> float:
    """Plain argmax — no decision layer at all.

    The calibration-free secondary read: if a verdict only holds under a fitted
    decision rule, it is a statement about the decision rule.
    """
    pred = np.asarray(np.argmax(proba_eval, axis=1), dtype=np.int64)
    return balanced_accuracy_fast(y_eval, pred, n_classes=proba_eval.shape[1])


def frozen_weight_layer(weights: NDArray[np.float64]) -> DecisionLayer:
    """Decision layer with weights fixed in advance (never re-fitted).

    Appropriate only for *single-arm perturbation* controls, where challenger
    and stock share a calibration by construction. For two-matrix comparisons
    use :func:`honest_weighted_layer`.
    """

    def layer(
        _y_fit: NDArray[np.int64],
        _proba_fit: NDArray[np.float64],
        y_eval: NDArray[np.int64],
        proba_eval: NDArray[np.float64],
    ) -> float:
        pred = np.asarray(np.argmax(proba_eval * weights, axis=1), dtype=np.int64)
        return balanced_accuracy_fast(y_eval, pred, n_classes=proba_eval.shape[1])

    return layer


@dataclass
class MechanismScreen:
    """Result of :func:`screen_mechanism`."""

    sizes: NDArray[np.int64]
    """Ensemble sizes M the delta was measured at."""
    deltas: NDArray[np.float64]
    """Mean challenger-minus-stock delta at each size."""
    stderr: NDArray[np.float64]
    """Standard error of each delta across evaluation cells."""
    exponent: float
    """OLS slope of ``log|delta|`` on ``log M``; NaN if the delta changes sign."""
    exponent_sd: float
    """Bootstrap standard deviation of the exponent over cells."""
    verdict: str
    """Human-readable classification, from :func:`classify_exponent`."""
    n_bootstrap_finite: int = 0
    """Resamples that yielded a defined exponent."""
    n_bootstrap: int = 0
    """Resamples attempted. When ``n_bootstrap_finite`` falls well short of this,
    many resamples produced a delta that crossed zero — ``exponent_sd`` is then
    not to be trusted, and neither is the exponent."""

    def __str__(self) -> str:
        sizes = "".join(f"{int(m):>12d}" for m in self.sizes)
        deltas = "".join(f"{d:>+12.6f}" for d in self.deltas)
        stderr = "".join(f"{s:>+12.6f}" for s in self.stderr)
        return (
            f"   M     {sizes}\n"
            f"   delta {deltas}\n"
            f"   se    {stderr}\n"
            f"   EXPONENT {self.exponent:+.3f} (bootstrap sd {self.exponent_sd:.3f}, "
            f"{self.n_bootstrap_finite}/{self.n_bootstrap} finite)\n"
            f"   VERDICT: {self.verdict}"
        )


def decay_exponent(
    sizes: Sequence[int] | NDArray[np.int64],
    deltas: Sequence[float] | NDArray[np.float64],
) -> float:
    """OLS slope of ``log|delta|`` on ``log M``.

    Returns NaN when any delta is non-positive: a delta that crosses zero has
    no exponent, and — importantly — is *not* evidence of information.

    Fit at least four sizes where you can. s6e7 published a ``+0.475`` from a
    three-point fit and had to correct it to ``-0.158`` at six points; three
    points do not constrain a slope whose per-point noise is comparable to its
    range.

    >>> import numpy as np
    >>> round(decay_exponent([1, 2, 4], [0.4, 0.2, 0.1]), 3)
    -1.0
    """
    delta_arr = np.asarray(deltas, dtype=np.float64)
    if len(delta_arr) < MIN_SIZES_FOR_SLOPE or np.any(delta_arr <= 0):
        return float("nan")
    size_arr = np.asarray(sizes, dtype=np.float64)
    return float(np.polyfit(np.log(size_arr), np.log(delta_arr), 1)[0])


def classify_exponent(
    exponent: float,
    *,
    null_band: tuple[float, float] | None = None,
) -> str:
    """Verdict for a measured exponent, by distance from the robust anchors.

    Cuts fall midway between the anchors (``-0.25`` and ``-0.75``), so each
    verdict names the nearest of :data:`INFORMATION_ANCHOR`,
    :data:`SELECTION_NOISE_ANCHOR` and :data:`VARIANCE_REDUCTION_ANCHOR`. Only
    the outer two anchors are robust across decision layers, so a negative that
    reaches the bottom is reported as "not information" rather than as variance
    reduction specifically.

    Args:
        exponent: The measured exponent.
        null_band: Optional ``(low, high)`` selection-noise band measured on
            your own data with :func:`selection_null_arms`. Anything inside it
            is indistinguishable from the winner's curse. The band is
            instrument-dependent — never import one from another competition.

    >>> classify_exponent(0.5) == INFORMATION
    True
    >>> classify_exponent(-1.1) == NOT_INFORMATION
    True
    """
    if math.isnan(exponent):
        return UNDEFINED
    if exponent > INFORMATION_ANCHOR:
        return INFORMATION
    if null_band is not None and null_band[0] <= exponent <= null_band[1]:
        return SELECTION_NOISE
    if exponent <= 0.5 * (SELECTION_NOISE_ANCHOR + VARIANCE_REDUCTION_ANCHOR):
        return NOT_INFORMATION
    if exponent <= 0.5 * (INFORMATION_ANCHOR + SELECTION_NOISE_ANCHOR):
        return SELECTION_NOISE
    return SLOW_DECAY


def _subsets(
    n_members: int, size: int, max_subsets: int, seed: int
) -> list[tuple[int, ...]]:
    """All ``C(n, size)`` member subsets, or a random sample if there are too many.

    Averaging over *subsets* rather than over orderings is exact and far
    cheaper: every ordering prefix of length m is a subset of size m, and each
    subset is scored once.
    """
    if math.comb(n_members, size) <= max_subsets:
        return list(itertools.combinations(range(n_members), size))
    rng = np.random.default_rng(seed)
    sampled: set[tuple[int, ...]] = set()
    while len(sampled) < max_subsets:
        sampled.add(tuple(sorted(rng.choice(n_members, size, replace=False).tolist())))
    return sorted(sampled)


def _cell(
    challenger: NDArray[np.float64],
    stock: NDArray[np.float64],
    y: NDArray[np.int64],
    fit_idx: NDArray[np.int64],
    eval_idx: NDArray[np.int64],
    max_size: int,
    layer: DecisionLayer,
    max_subsets: int,
    seed: int,
) -> NDArray[np.float64]:
    """Paired delta at every M for one (fit, eval) split."""
    out = np.zeros(max_size)
    for size in range(1, max_size + 1):
        subsets = _subsets(len(challenger), size, max_subsets, seed)
        total = 0.0
        for idx in subsets:
            members = list(idx)
            c_mean = challenger[members].mean(axis=0)
            s_mean = stock[members].mean(axis=0)
            total += layer(
                y[fit_idx], c_mean[fit_idx], y[eval_idx], c_mean[eval_idx]
            ) - layer(y[fit_idx], s_mean[fit_idx], y[eval_idx], s_mean[eval_idx])
        out[size - 1] = total / len(subsets)
    return out


def screen_mechanism(
    challenger: NDArray[np.float64],
    stock: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    max_size: int | None = None,
    layer: DecisionLayer | None = None,
    n_splits: int = 6,
    n_bootstrap: int = 400,
    max_subsets: int = 70,
    null_band: tuple[float, float] | None = None,
    seed: int = 0,
    n_jobs: int = 1,
) -> MechanismScreen:
    """Measure a lever's decay exponent: information, selection noise, or noise removal.

    Both arms must be *paired* — challenger member ``i`` and stock member ``i``
    differ only by the lever (same seed, same folds) — and each average over
    the same subset of members at every size. The delta is read on
    ``2 * n_splits`` stratified half-splits, each direction, so the decision
    layer is fitted and scored on disjoint rows.

    Args:
        challenger: ``(n_members, n_rows, n_classes)`` per-member probabilities
            for the arm under test.
        stock: Same shape, the incumbent arm.
        y: Encoded ground-truth labels, ``n_rows`` long, in the same row order.
        max_size: Largest M to measure, default all members. Prefer at least 4
            — see :func:`decay_exponent`.
        layer: Decision layer, default :func:`honest_weighted_layer`.
        n_splits: Stratified half-splits; each contributes two cells.
        n_bootstrap: Resamples of the cells for the exponent's standard
            deviation.
        max_subsets: Cap on subsets averaged per size; above it, subsets are
            sampled (documented rather than silent, since a cap bounds
            coverage).
        null_band: Selection-noise band measured on your own data with
            :func:`selection_null_arms`, passed straight to
            :func:`classify_exponent`. Leaving it ``None`` means the verdict
            rests on the two robust anchors alone and cannot say "this is
            indistinguishable from the winner's curse" — measure the band once
            per competition and pass it to every screen.
        seed: RNG seed for splits, subset sampling and the bootstrap.
        n_jobs: Cells to evaluate in parallel (needs ``joblib``, which ships
            with scikit-learn). ``1`` runs in-process.

    Returns:
        A :class:`MechanismScreen`; ``print`` it for the report table.
    """
    if challenger.shape != stock.shape:
        msg = f"arm shapes differ: {challenger.shape} vs {stock.shape}"
        raise ValueError(msg)
    if challenger.shape[1] != len(y):
        msg = f"row mismatch: arms have {challenger.shape[1]} rows, y has {len(y)}"
        raise ValueError(msg)
    n_members = len(challenger)
    size_max = n_members if max_size is None else min(max_size, n_members)
    decide = honest_weighted_layer if layer is None else layer

    cells: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for split in range(n_splits):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed + split
        )
        left, right = next(splitter.split(y, y))
        cells += [(left, right), (right, left)]

    args = [
        (challenger, stock, y, fit, ev, size_max, decide, max_subsets, seed)
        for fit, ev in cells
    ]
    if n_jobs == 1:
        rows = [_cell(*a) for a in args]
    else:
        from joblib import Parallel, delayed  # noqa: PLC0415

        rows = list(Parallel(n_jobs=n_jobs)(delayed(_cell)(*a) for a in args))
    measured = np.asarray(rows, dtype=np.float64)

    sizes = np.arange(1, size_max + 1, dtype=np.int64)
    deltas = measured.mean(axis=0)
    stderr = measured.std(axis=0) / np.sqrt(len(measured))
    exponent = decay_exponent(sizes, deltas)

    rng = np.random.default_rng(seed)
    boots = np.asarray(
        [
            decay_exponent(
                sizes,
                measured[rng.integers(0, len(measured), len(measured))].mean(axis=0),
            )
            for _ in range(n_bootstrap)
        ]
    )
    finite = boots[np.isfinite(boots)]
    exponent_sd = float(finite.std()) if len(finite) else float("nan")
    return MechanismScreen(
        sizes=sizes,
        deltas=deltas,
        stderr=stderr,
        exponent=exponent,
        exponent_sd=exponent_sd,
        verdict=classify_exponent(exponent, null_band=null_band),
        n_bootstrap_finite=len(finite),
        n_bootstrap=n_bootstrap,
    )


def mean_shift_control(
    members: NDArray[np.float64], y: NDArray[np.int64], *, lam: float
) -> NDArray[np.float64]:
    """Known-real control (C1): mix every member toward ``onehot(y)``.

    A uniform shift toward the truth is information by construction, and reads
    at the top of the scale (s6e7: ``-0.027``, against a prediction of 0).

    ⚠️ **Consumes the labels — quarantined instrument calibration. This can
    never become a lever.** Keep it out of any code path that produces a
    submission.

    One consequence worth checking on your own data: mixing toward ``onehot(y)``
    adds mass in proportion to class frequency, so it shifts the optimal
    decision weights too. Re-read the control under
    :func:`argmax_layer` as well; if the anchor only holds under a fitted
    decision rule, the whole scale is an artifact of that rule.
    """
    onehot = np.eye(members.shape[2], dtype=np.float64)[y]
    return np.asarray((1.0 - lam) * members + lam * onehot, dtype=np.float64)


def variance_reduction_control(
    members: NDArray[np.float64],
    target_members: NDArray[np.float64],
    *,
    lam: float,
) -> NDArray[np.float64]:
    """Known-fake control (C3): shrink each member toward a **disjoint** average.

    Same mean, lower variance — the definition of a variance reducer, and it
    reads at the bottom of the scale (s6e7: ``-1.002``, against a prediction of
    -1).

    ``target_members`` must not contain any member of ``members``: shrinking
    toward an average that includes the member forces the delta to zero at full
    M by construction and fakes the result. Split your per-member arrays into
    two disjoint halves and pass one as each argument.

    A related trap, from a prediction that failed: shrinking toward a *better*
    posterior (the deployed stack) is not a free label-free mean-shift control.
    A stack is itself an average over many fits, so "shrink toward something
    better" is mechanically a variance reducer — it read ``-1.086``.
    """
    return np.asarray(
        (1.0 - lam) * members + lam * target_members.mean(axis=0), dtype=np.float64
    )


def selection_null_arms(
    members: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    seed: int = 0,
    layer: DecisionLayer | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Known-fake control (C2): two disjoint halves of one leg, oriented by the M=1 read.

    Both halves are the same model — any delta between them is pure noise. By
    choosing *which* half to call the challenger from its M=1 score, this
    reproduces the winner's curse that every real screen performs, so screening
    the returned pair over many seeds gives the **null distribution** of the
    exponent on your data (and hence the instrument's minimum detectable
    effect).

    Pass the resulting band to :func:`classify_exponent`; do not import a band
    measured elsewhere, since it depends strongly on the decision layer.

    Returns:
        ``(challenger, stock)``, disjoint halves of ``members``.
    """
    decide = honest_weighted_layer if layer is None else layer
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(members))
    half = len(members) // 2
    left, right = members[order[:half]], members[order[half : 2 * half]]
    scores = [
        float(np.mean([decide(y, arm[i], y, arm[i]) for i in range(half)]))
        for arm in (left, right)
    ]
    return (left, right) if scores[0] >= scores[1] else (right, left)


def calibrate_lambda(
    score_at: Callable[[float], float],
    target: float,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    n_iter: int = 18,
) -> float:
    """Bisect a control's mixing weight so it reads ``target`` at M=1.

    Controls must be **magnitude-matched** to the lever under test: an exponent
    difference between a +0.0009 control and a +0.0003 lever is confounded with
    size. ``score_at`` maps a mixing weight to the M=1 delta and is assumed
    monotonically increasing in it.

    >>> round(calibrate_lambda(lambda x: 2 * x, 0.5), 3)
    0.25
    """
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if score_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
