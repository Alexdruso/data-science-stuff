"""Imbalance-aware classification losses (PyTorch).

The headline lever here is *logit adjustment* (balanced softmax): during
training add ``tau * log(prior_c)`` to each class logit before cross-entropy,
then predict with the RAW logits. This shifts the decision boundary toward
rare classes without distorting inference probabilities — on s6e6 it was the
difference between NNs plateauing at ~0.95 and RealMLP's 0.9688 (neither
class weights nor oversampling closed that gap).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def smooth_ce_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    ls: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    loss_prob_multipliers: torch.Tensor | None = None,
    per_sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Label-smoothed cross-entropy on *probabilities*, with imbalance knobs.

    Note: ``y_pred`` is expected to be post-softmax probabilities (this is
    the RealMLP convention), not logits.

    Args:
        y_true: (n,) integer class labels.
        y_pred: (n, n_classes) predicted probabilities.
        ls: Label-smoothing epsilon.
        class_weights: Optional (n_classes,) per-class loss weights.
        focal_gamma: Focal-loss exponent; 0 disables focusing.
        loss_prob_multipliers: Optional (n_classes,) multipliers applied to
            the probabilities inside the loss then renormalized — the
            balanced-softmax / logit-adjustment term expressed on
            probabilities (``class_counts_geonormed ** prior_power``).
        per_sample_weight: Optional (n,) per-sample loss weights.

    Returns:
        Scalar loss tensor.
    """
    n_classes = y_pred.size(1)
    if loss_prob_multipliers is not None:
        y_pred = y_pred * loss_prob_multipliers[None, :]
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-15)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if focal_gamma > 0:
        pt = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1).clamp(1e-15, 1.0)
        per_sample_loss = per_sample_loss * torch.pow(1.0 - pt, focal_gamma)
    sample_weights = None
    if class_weights is not None:
        sample_weights = class_weights[y_true]
    if per_sample_weight is not None:
        sample_weights = (
            per_sample_weight
            if sample_weights is None
            else sample_weights * per_sample_weight
        )
    if sample_weights is not None:
        return (
            per_sample_loss * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-15)
    return per_sample_loss.mean()


def logit_adjustment(class_counts: NDArray[Any], *, tau: float = 1.0) -> torch.Tensor:
    """Logit-adjustment offsets: ``tau * (log(counts) - mean(log(counts)))``.

    Add the returned tensor to the model's logits inside the training loss
    (``criterion(model(x) + adj, y)``) and predict with raw logits. The
    geometric-mean normalization keeps the offsets mean-zero, so only the
    *relative* priors shift the boundary. ``tau`` slightly above 1 (s6e6
    used 1.075) over-corrects toward rare classes.

    >>> import numpy as np
    >>> logit_adjustment(np.array([100, 100])).tolist()
    [0.0, 0.0]

    Args:
        class_counts: (n_classes,) training-split class counts (from the
            fold's training rows only — priors are fold statistics too).
        tau: Adjustment strength.

    Returns:
        (n_classes,) float32 CPU tensor; move to the model's device with
        ``.to(device)``.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    log_prior = np.log(counts) - np.log(counts).mean()
    return torch.tensor(tau * log_prior, dtype=torch.float32)
