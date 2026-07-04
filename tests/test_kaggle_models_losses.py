"""Tests for data_science_stuff.kaggle.models.losses."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from data_science_stuff.kaggle.models.losses import logit_adjustment, smooth_ce_loss


def _fixture() -> "tuple[torch.Tensor, torch.Tensor]":
    torch.manual_seed(0)
    logits = torch.randn(64, 3)
    proba = torch.softmax(logits, dim=1)
    y = torch.randint(0, 3, (64,))
    return y, proba


def test_smooth_ce_loss_matches_cross_entropy_with_knobs_off() -> None:
    y, proba = _fixture()
    ours = smooth_ce_loss(y, proba)
    reference = F.nll_loss(torch.log(proba), y)
    assert ours.item() == pytest.approx(reference.item(), abs=1e-6)


def test_smooth_ce_loss_each_knob_changes_the_value() -> None:
    y, proba = _fixture()
    base = smooth_ce_loss(y, proba).item()

    smoothed = smooth_ce_loss(y, proba, ls=0.1).item()
    focal = smooth_ce_loss(y, proba, focal_gamma=2.0).item()
    weighted = smooth_ce_loss(
        y, proba, class_weights=torch.tensor([1.0, 2.0, 4.0])
    ).item()
    adjusted = smooth_ce_loss(
        y, proba, loss_prob_multipliers=torch.tensor([2.0, 1.0, 0.5])
    ).item()

    assert smoothed != pytest.approx(base)
    assert focal < base  # focusing down-weights easy examples
    assert weighted != pytest.approx(base)
    assert adjusted != pytest.approx(base)


def test_smooth_ce_loss_per_sample_weight_composes_with_class_weights() -> None:
    y, proba = _fixture()
    psw = torch.rand(len(y)) + 0.5
    cw = torch.tensor([1.0, 2.0, 4.0])

    combined = smooth_ce_loss(y, proba, class_weights=cw, per_sample_weight=psw)
    solo = smooth_ce_loss(y, proba, per_sample_weight=psw)

    assert torch.isfinite(combined)
    assert combined.item() != pytest.approx(solo.item())


def test_logit_adjustment_closed_form_and_mean_zero() -> None:
    counts = np.array([700, 200, 100])
    adj = logit_adjustment(counts, tau=1.075)

    log_prior = np.log(counts) - np.log(counts).mean()
    np.testing.assert_allclose(adj.numpy(), 1.075 * log_prior, rtol=1e-6)
    assert adj.mean().item() == pytest.approx(0.0, abs=1e-6)
    # Majority class gets a positive offset (handicapped in the loss),
    # rare class a negative one (boosted).
    assert adj[0] > 0 > adj[2]
