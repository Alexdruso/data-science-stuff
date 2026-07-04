"""Ensemble blending: Nelder-Mead weight search and diversity diagnostics.

Unifies the ``ensemble.py`` blend-weight optimizers used by s6e5 (clip
normalization, AUC) and s6e6 (softmax normalization, balanced accuracy on
argmax): same optimizer and tolerances, metric and normalization are
parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import recall_score

# score_fn(y_true, blended) -> float, higher is better. Close your metric over
# any argmax/proba handling, e.g.
# ``lambda y, b: balanced_accuracy_score(y, b.argmax(axis=1))``.
ScoreFn = Callable[[NDArray[Any], NDArray[np.float64]], float]
Normalize = Literal["clip", "softmax"]

_MIN_MODELS = 2


def normalize_weights(
    raw: NDArray[np.float64], *, method: Normalize
) -> NDArray[np.float64]:
    """Map an unconstrained weight vector onto the probability simplex.

    ``"clip"`` clips negatives to zero then rescales to sum 1 (weights stay
    interpretable as raw magnitudes, and can be exactly zero); ``"softmax"``
    exponentiates (all weights strictly positive, the optimizer works in an
    unconstrained log space).

    >>> import numpy as np
    >>> normalize_weights(np.array([1.0, 1.0]), method="clip").tolist()
    [0.5, 0.5]
    >>> normalize_weights(np.array([-1.0, 3.0]), method="clip").tolist()
    [0.0, 1.0]
    >>> normalize_weights(np.array([0.0, 0.0]), method="softmax").tolist()
    [0.5, 0.5]
    """
    if method == "clip":
        w = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
        total = float(w.sum())
        if total <= 0.0:
            msg = "all weights clipped to zero"
            raise ValueError(msg)
        return np.asarray(w / total, dtype=np.float64)
    e = np.exp(np.asarray(raw, dtype=np.float64) - float(np.max(raw)))
    return np.asarray(e / e.sum(), dtype=np.float64)


def blend(
    arrays: Sequence[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weighted sum of prediction arrays (weights should already be normalized)."""
    out = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for w, a in zip(weights, arrays):
        out += float(w) * a
    return out


def optimize_blend_weights(
    oof_list: Sequence[NDArray[np.float64]],
    y: NDArray[Any],
    score_fn: ScoreFn,
    *,
    normalize: Normalize = "clip",
    x0: NDArray[np.float64] | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-6,
    fatol: float = 1e-6,
) -> NDArray[np.float64]:
    """Nelder-Mead search for blend weights maximizing ``score_fn`` on OOF.

    Works for 1-D predictions (binary proba / regression) and 2-D multiclass
    probability arrays alike — the metric handling lives in ``score_fn``.
    Conditional blending (separate weights per data regime) is the caller's
    job: pre-slice the arrays per mask and optimize each slice.

    Args:
        oof_list: One OOF prediction array per model, all aligned with ``y``.
        y: Ground-truth labels/values in the same (sorted) row order.
        score_fn: ``(y, blended) -> float``, higher is better.
        normalize: Weight parameterization, see :func:`normalize_weights`.
        x0: Optional raw starting point (defaults to uniform weights).
        maxiter: Nelder-Mead iteration cap.
        xatol: Nelder-Mead absolute tolerance on the simplex.
        fatol: Nelder-Mead absolute tolerance on the objective.

    Returns:
        (n_models,) normalized weights summing to 1, in ``oof_list`` order.
    """
    n_models = len(oof_list)
    if n_models < _MIN_MODELS:
        msg = "need at least two models to blend"
        raise ValueError(msg)

    def neg_score(raw: NDArray[np.float64]) -> float:
        return -float(
            score_fn(y, blend(oof_list, normalize_weights(raw, method=normalize)))
        )

    start = np.ones(n_models) if x0 is None else np.asarray(x0, dtype=np.float64)
    result = minimize(
        neg_score,
        start,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol},
    )
    return normalize_weights(np.asarray(result.x, dtype=np.float64), method=normalize)


def diversity_report(
    oof_arrays: Mapping[str, NDArray[np.float64]],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    *,
    anchor: str,
) -> pd.DataFrame:
    """Per-class recall and error overlap of each model versus an anchor.

    Global probability correlation does not predict metric lift; what matters
    is whether a model fixes *different* errors. ``fixes_anchor`` is the
    fraction of rows the anchor gets wrong that this model gets right;
    ``anchor_fixes`` is the reverse. The anchor's own overlap columns are NaN.

    Args:
        oof_arrays: ``{model_name: (n, n_classes) OOF probabilities}``.
        y: Encoded ground-truth labels.
        class_names: Class names in encoded order.
        anchor: Key in ``oof_arrays`` to compare everything against.

    Returns:
        DataFrame indexed by model name with columns ``recall_<class>`` per
        class plus ``fixes_anchor`` and ``anchor_fixes``.
    """
    if anchor not in oof_arrays:
        msg = f"anchor {anchor!r} not among models"
        raise ValueError(msg)
    class_ids = list(range(len(class_names)))
    anchor_pred = np.argmax(oof_arrays[anchor], axis=1)
    anchor_wrong = anchor_pred != y

    rows: dict[str, dict[str, float]] = {}
    for model, arr in oof_arrays.items():
        pred = np.argmax(arr, axis=1)
        recalls = recall_score(y, pred, average=None, labels=class_ids)
        row = {f"recall_{c}": float(r) for c, r in zip(class_names, recalls)}
        if model == anchor:
            row["fixes_anchor"] = float("nan")
            row["anchor_fixes"] = float("nan")
        else:
            wrong = pred != y
            row["fixes_anchor"] = float((pred[anchor_wrong] == y[anchor_wrong]).mean())
            row["anchor_fixes"] = float((anchor_pred[wrong] == y[wrong]).mean())
        rows[model] = row
    report: pd.DataFrame = pd.DataFrame.from_dict(rows, orient="index")
    return report
