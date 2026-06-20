"""Post-processing utilities for PS S6E6 training scripts.

Shared between all train_*.py scripts — do not add model-specific logic here.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score


def optimize_thresholds(
    oof_proba: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 15,
) -> tuple[np.ndarray, float]:
    """Find per-class scale weights that maximise balanced_acc on OOF proba.

    Returns (weights, best_score). Apply via argmax(oof_proba * weights).
    3 params on 500k+ samples → negligible overfitting risk.
    """
    base_score = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))

    def neg_ba(log_w: np.ndarray) -> float:
        w = np.exp(log_w)
        return -float(balanced_accuracy_score(y, np.argmax(oof_proba * w, axis=1)))

    best_score = base_score
    best_x = np.zeros(oof_proba.shape[1])

    for seed in range(n_restarts):
        rng = np.random.default_rng(seed)
        result = minimize(
            neg_ba,
            rng.normal(0, 0.3, size=oof_proba.shape[1]),
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_x = result.x

    return np.exp(best_x), best_score


def save_threshold_weights(
    weights: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    with path.open("w") as f:
        json.dump(dict(zip(class_names, weights.tolist())), f, indent=2)
