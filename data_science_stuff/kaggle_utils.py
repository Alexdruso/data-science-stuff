"""Shared utilities for Kaggle Playground Series competitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

# Per-class weights accepted either as a plain sequence or as the float array
# returned by :func:`tune_decision_weights`.
Weights = Union[Sequence[float], NDArray[np.float64]]


def weighted_predict(
    proba: NDArray[np.float64],
    weights: Weights,
) -> NDArray[np.int64]:
    """Argmax of class probabilities scaled by per-class weights.

    Args:
        proba: (n_samples, n_classes) predicted probabilities.
        weights: Per-class multiplicative weights, length n_classes.

    Returns:
        (n_samples,) integer class predictions.

    >>> import numpy as np
    >>> p = np.array([[0.6, 0.4], [0.55, 0.45]])
    >>> weighted_predict(p, [1.0, 1.0]).tolist()
    [0, 0]
    >>> weighted_predict(p, [1.0, 2.0]).tolist()
    [1, 1]
    """
    w = np.asarray(weights, dtype=np.float64)
    return np.asarray(np.argmax(proba * w, axis=1), dtype=np.int64)


def tune_decision_weights(
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    *,
    seed: int = 42,
    n_restarts: int = 15,
    maxiter: int = 3000,
    xatol: float = 1e-7,
    fatol: float = 1e-7,
    init_scale: float = 0.3,
    score_fn: Callable[[NDArray[np.int64], NDArray[np.int64]], float] | None = None,
) -> NDArray[np.float64]:
    """Find per-class weights maximizing the score of argmax(proba * w).

    Useful when the training objective (e.g. multiclass log-loss) is misaligned
    with a balanced-accuracy metric on imbalanced classes: plain argmax defaults
    to the majority prior, suppressing recall on rare classes. Optimizing
    per-class multiplicative weights re-balances the decision rule without
    retraining.

    Optimization runs in log-weight space with class 0 fixed at log-weight 0
    (the problem is scale-invariant, so K-1 free degrees of freedom), via
    Nelder-Mead with several random restarts for robustness. Defaults follow
    the strongest settings proven on s6e6 (15 restarts, 1e-7 tolerances).

    Args:
        y_true: (n_samples,) integer class labels in [0, n_classes).
        proba: (n_samples, n_classes) predicted probabilities.
        seed: RNG seed for restart initial points.
        n_restarts: Number of restarts (the first starts at equal weights).
        maxiter: Nelder-Mead iteration cap per restart.
        xatol: Nelder-Mead absolute tolerance on the simplex.
        fatol: Nelder-Mead absolute tolerance on the objective.
        init_scale: Std-dev of the random restart initial log-weights.
        score_fn: ``(y_true, y_pred) -> float``, higher is better; defaults
            to balanced accuracy.

    Returns:
        (n_classes,) weight vector normalized so the max weight is 1.0.
    """
    n_classes = proba.shape[1]
    rng = np.random.default_rng(seed)
    metric: Callable[[NDArray[np.int64], NDArray[np.int64]], float] = (
        score_fn
        if score_fn is not None
        else lambda yt, yp: float(balanced_accuracy_score(yt, yp))
    )

    def neg_score(log_w_free: NDArray[np.float64]) -> float:
        log_w = np.concatenate([[0.0], log_w_free])
        pred = np.asarray(np.argmax(proba * np.exp(log_w), axis=1), dtype=np.int64)
        return -float(metric(y_true, pred))

    best_log_free = np.zeros(n_classes - 1)
    best_score = neg_score(best_log_free)
    for i in range(n_restarts):
        x0 = (
            np.zeros(n_classes - 1)
            if i == 0
            else rng.normal(0.0, init_scale, n_classes - 1)
        )
        res = minimize(
            neg_score,
            x0,
            method="Nelder-Mead",
            options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter},
        )
        if float(res.fun) < best_score:
            best_score = float(res.fun)
            best_log_free = res.x

    w = np.exp(np.concatenate([[0.0], best_log_free]))
    return np.asarray(w / w.max(), dtype=np.float64)


def classification_report_dict(
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    *,
    weights: Weights | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Per-class recall + balanced accuracy + confusion for argmax (and weights).

    Args:
        y_true: (n_samples,) integer labels.
        proba: (n_samples, n_classes) probabilities.
        weights: Optional per-class weights; if given, the weighted decision
            rule is reported alongside plain argmax.
        labels: Optional class names used as keys in the per-class recall dicts.

    Returns:
        Dict with keys: ``argmax_balanced_acc``, ``per_class_recall_argmax``,
        ``confusion_argmax`` and, when ``weights`` is given, the matching
        ``weighted_*`` keys.
    """
    n_classes = proba.shape[1]
    names = list(labels) if labels is not None else [str(i) for i in range(n_classes)]
    class_ids = list(range(n_classes))

    def _recall(pred: NDArray[np.int64]) -> dict[str, float]:
        rec = recall_score(y_true, pred, average=None, labels=class_ids)
        return {name: float(v) for name, v in zip(names, rec)}

    argmax_pred = np.asarray(np.argmax(proba, axis=1), dtype=np.int64)
    out: dict[str, object] = {
        "argmax_balanced_acc": float(balanced_accuracy_score(y_true, argmax_pred)),
        "per_class_recall_argmax": _recall(argmax_pred),
        "confusion_argmax": confusion_matrix(y_true, argmax_pred),
    }
    if weights is not None:
        w_pred = weighted_predict(proba, weights)
        out["weighted_balanced_acc"] = float(balanced_accuracy_score(y_true, w_pred))
        out["per_class_recall_weighted"] = _recall(w_pred)
        out["confusion_weighted"] = confusion_matrix(y_true, w_pred)
    return out


def print_classification_report(report: dict[str, object]) -> None:
    """Pretty-print the dict returned by :func:`classification_report_dict`."""

    def _fmt(recalls: dict[str, float]) -> str:
        return "  ".join(f"{k}={v:.4f}" for k, v in recalls.items())

    argmax_acc = cast(float, report["argmax_balanced_acc"])
    argmax_rec = cast("dict[str, float]", report["per_class_recall_argmax"])
    print(f"argmax   balanced_acc: {argmax_acc:.4f}")
    print(f"  per-class recall (argmax):   {_fmt(argmax_rec)}")
    if "weighted_balanced_acc" in report:
        weighted_acc = cast(float, report["weighted_balanced_acc"])
        weighted_rec = cast("dict[str, float]", report["per_class_recall_weighted"])
        print(f"weighted balanced_acc: {weighted_acc:.4f}")
        print(f"  per-class recall (weighted): {_fmt(weighted_rec)}")


def save_cv_result(
    results_dir: Path,
    model: str,
    fold_scores: list[float],
    oof_score: float,
    metric_name: str = "score",
) -> None:
    """Append a CV result row to results_dir/cv_scores.csv.

    Args:
        results_dir: Directory where cv_scores.csv lives (created if absent).
        model: Identifier string for this run (e.g. "baseline_lgbm_v1").
        fold_scores: Per-fold metric values.
        oof_score: Overall out-of-fold metric value.
        metric_name: Column suffix, e.g. "auc" → oof_auc / fold_1_auc.
    """
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "cv_scores.csv"

    row: dict[str, object] = {
        "model": model,
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        f"oof_{metric_name}": round(oof_score, 6),
    }
    for i, score in enumerate(fold_scores, 1):
        row[f"fold_{i}_{metric_name}"] = round(score, 6)

    new_row = pd.DataFrame([row])
    if path.exists():
        pd.concat([pd.read_csv(path), new_row], ignore_index=True).to_csv(
            path, index=False
        )
    else:
        new_row.to_csv(path, index=False)

    print(f"CV result appended → {path}")
