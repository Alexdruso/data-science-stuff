"""Comparing experiment arms: seed-paired deltas and honest split-half blend reads.

Two reads every campaign rebuilt by hand:

- :func:`paired_seed_deltas` (s6e8 ``run_arm.compare``): each arm's score
  minus the baseline's **at the same seed**. Pairing cancels the seed-to-seed
  noise shared by both arms, which is usually much larger than the effect.
- :func:`honest_blend_score` / :func:`honest_blend_delta` (s6e10
  ``blend_check``): what a candidate leg adds to a base blend when the blend
  weight is fit on one half of the rows and scored on the other. A weight fit
  and scored on the same rows always finds *some* gain.

These answer "is the delta there at all". To decide *why* a delta exists
(information vs selection noise vs variance reduction) use
:func:`data_science_stuff.kaggle.screening.screen_mechanism`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from data_science_stuff.kaggle.blending import ScoreFn


def paired_seed_deltas(
    scores: Mapping[str, Mapping[Any, float]], baseline: str
) -> pd.DataFrame:
    """Per-arm mean score and seed-paired delta versus ``baseline``.

    Args:
        scores: ``{arm: {seed: score}}`` (e.g. a ``results/arm_scores.json``).
        baseline: The arm every other arm is compared against.

    Returns:
        One row per arm (sorted by ``delta``, best first) with ``n_seeds``
        (seeds shared with the baseline), ``mean`` (over all of the arm's
        seeds), ``delta`` (mean paired delta), ``sd_delta``, ``se_delta``
        and ``t`` (``delta / se_delta``). Delta columns are NaN for the
        baseline itself and for arms sharing no seed with it.

    >>> s = {"base": {1: 0.90, 2: 0.92}, "new": {1: 0.91, 2: 0.93}}
    >>> round(float(paired_seed_deltas(s, "base").loc["new", "delta"]), 6)
    0.01
    """
    if baseline not in scores:
        msg = f"baseline arm {baseline!r} not in scores"
        raise KeyError(msg)
    base = scores[baseline]
    rows: list[dict[str, Any]] = []
    for arm, per_seed in scores.items():
        shared = [s for s in per_seed if s in base] if arm != baseline else []
        deltas = np.array([per_seed[s] - base[s] for s in shared], dtype=float)
        n = len(deltas)
        sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        delta = float(deltas.mean()) if n else float("nan")
        rows.append(
            {
                "arm": arm,
                "n_seeds": n,
                "mean": float(np.mean(list(per_seed.values()))),
                "delta": delta,
                "sd_delta": sd,
                "se_delta": se,
                "t": delta / se if n > 1 and se > 0 else float("nan"),
            }
        )
    out = pd.DataFrame(rows).set_index("arm")
    return out.sort_values("delta", ascending=False, na_position="last")


class HonestBlend(NamedTuple):
    """Held-out score of ``(1 - w) * base + w * candidate`` and the fitted ``w``s."""

    score: float
    weights: list[float]
    half_scores: list[float]


def _default_auc(y: NDArray[Any], p: NDArray[np.float64]) -> float:
    return float(roc_auc_score(y, p))


def honest_blend_score(
    y: NDArray[Any],
    base: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    score_fn: ScoreFn | None = None,
    grid: Sequence[float] | None = None,
    seeds: Sequence[int] = (0,),
) -> HonestBlend:
    """Score a two-way blend whose weight is fit on the other half of the rows.

    For each seed the rows are split into two random halves; the weight is
    picked on one half (grid search) and scored on the other, in both
    directions. Rank-transform AUC inputs first if the two arrays live on
    different scales.

    Args:
        y: Targets.
        base: Base predictions (e.g. the current best blend's OOF).
        candidate: The leg being considered.
        score_fn: ``(y, pred) -> float``, higher is better; default ROC AUC.
        grid: Candidate weights; default ``0, 0.05, ..., 1``.
        seeds: Split seeds; more seeds average out the split noise.

    Returns:
        :class:`HonestBlend` — mean held-out score over all halves.
    """
    score = score_fn if score_fn is not None else _default_auc
    ws = np.linspace(0.0, 1.0, 21) if grid is None else np.asarray(grid, float)
    weights: list[float] = []
    half_scores: list[float] = []
    for seed in seeds:
        first = np.random.default_rng(seed).permutation(len(y)) < len(y) // 2
        for fit_mask in (first, ~first):
            eval_mask = ~fit_mask
            fit_scores = [
                score(y[fit_mask], (1 - w) * base[fit_mask] + w * candidate[fit_mask])
                for w in ws
            ]
            w_best = float(ws[int(np.argmax(fit_scores))])
            weights.append(w_best)
            blended = (1 - w_best) * base[eval_mask] + w_best * candidate[eval_mask]
            half_scores.append(float(score(y[eval_mask], blended)))
    return HonestBlend(float(np.mean(half_scores)), weights, half_scores)


def honest_blend_delta(
    y: NDArray[Any],
    base: NDArray[np.float64],
    new: NDArray[np.float64],
    old: NDArray[np.float64],
    *,
    score_fn: ScoreFn | None = None,
    grid: Sequence[float] | None = None,
    seeds: Sequence[int] = (0,),
) -> float:
    """Honest blend score with ``new`` minus with ``old``, on identical halves.

    The read for "does replacing leg ``old`` by ``new`` help the blend": both
    candidates see exactly the same splits, so the split noise cancels.
    Compare the result with a bar written down before running.
    """
    kwargs: dict[str, Any] = {"score_fn": score_fn, "grid": grid, "seeds": seeds}
    with_new = honest_blend_score(y, base, new, **kwargs)
    with_old = honest_blend_score(y, base, old, **kwargs)
    return with_new.score - with_old.score
