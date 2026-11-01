"""Tests for data_science_stuff.kaggle.models.realmlp (fast CPU smoke)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from data_science_stuff.kaggle.models.realmlp import (
    REALMLP_TD_CONFIG,
    NumericalPreprocessor,
    RealMLP_TD_Classifier,
    apply_schedule,
)

_TINY = {
    "n_ens": 2,
    "hidden_dims": [16],
    "epochs": 2,
    "train_bs": 64,
    "eval_bs": 256,
    "device": "cpu",
    "verbosity": 0,
    "pbld_hidden_dim": 4,
    "pbld_out_dim": 3,
}


@pytest.mark.parametrize(
    ("sched", "at_zero", "at_one"),
    [
        ("constant", 1.0, 1.0),
        ("cos", 1.0, 0.0),
        ("flat_cos", 1.0, 0.0),
        ("flat_anneal", 1.0, 0.0),
        ("sqrt_cos", 1.0, 0.0),
        ("expm4t", 1.0, np.exp(-4)),
    ],
)
def test_apply_schedule_endpoints(sched: str, at_zero: float, at_one: float) -> None:
    assert apply_schedule(1.0, 0.0, sched) == pytest.approx(at_zero)
    assert apply_schedule(1.0, 1.0, sched) == pytest.approx(at_one, abs=1e-12)


def test_apply_schedule_flat_phase_and_unknown() -> None:
    assert apply_schedule(1.0, 0.2, "flat_cos", flat_ratio=0.3) == 1.0
    assert apply_schedule(1.0, 0.2, "flat_anneal", flat_ratio=0.3) == 1.0
    with pytest.raises(ValueError, match="Unknown schedule"):
        apply_schedule(1.0, 0.5, "bogus")


def test_numerical_preprocessor_median_center_and_robust_scale() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, (100, 3)).astype(np.float32)
    pre = NumericalPreprocessor(["median_center", "robust_scale"]).fit(x)
    out = pre.transform(x)
    np.testing.assert_allclose(np.median(out, axis=0), 0.0, atol=1e-5)

    clipped = NumericalPreprocessor(["smooth_clip"]).fit(x).transform(x * 100)
    assert (np.abs(clipped) < 3.0).all()  # smooth clip bounds at ±3

    normed = NumericalPreprocessor(["l2_normalize"]).fit(x).transform(x)
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, rtol=1e-5)


def _frame(n: int, seed: int) -> "tuple[pd.DataFrame, NDArray[np.int64]]":
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1, 2], dtype=np.int64), n // 3 + 1)[:n]
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n) + y,
            "num_b": rng.normal(0, 1, n),
            "cat_small": rng.integers(0, 3, n),  # one-hot path (≤ onehot_thresh)
            "cat_large": rng.integers(0, 15, n),  # embedding path (> onehot_thresh)
        }
    )
    return df, y


def test_realmlp_td_classifier_smoke_fit_predict(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _frame(180, seed=0)
    x_va, y_va = _frame(60, seed=1)
    x_te, _ = _frame(30, seed=2)

    clf = RealMLP_TD_Classifier(**_TINY)
    clf.fit(
        x_tr,
        y_tr,
        x_va,
        y_va,
        cat_col_names=["cat_small", "cat_large"],
        ckpt_path=tmp_path / "ckpt.pth",
        X_test=x_te,
    )

    proba = clf.predict_proba(x_te)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(clf.predict(x_te)) <= {0, 1, 2}
    assert clf.best_score_ > 0
    assert clf.best_val_probs_ is not None
    assert clf.best_val_probs_.shape == (60, 3)


def test_realmlp_td_classifier_optional_branches(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)

    overrides = {
        **_TINY,
        "verbosity": 2,
        "class_weight_power": 0.5,
        "class_weight_multipliers": [1.0, 1.0, 2.0],
        "sample_weight_power": 0.5,
        "numeric_noise_std": 0.01,
        "focal_gamma": 1.0,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 5,
    }
    clf = RealMLP_TD_Classifier(**overrides)
    clf.fit(
        x_tr,
        y_tr,
        x_va,
        y_va,
        cat_col_names=["cat_small", "cat_large"],
        ckpt_path=tmp_path / "ckpt.pth",
        per_sample_weight=np.ones(len(y_tr)),
    )
    assert clf.predict_proba(x_va).shape == (45, 3)


def test_realmlp_td_classifier_numeric_only_and_no_ema(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)
    numeric = ["num_a", "num_b"]

    clf = RealMLP_TD_Classifier(**_TINY, ema_decay=0.0, loss_prior_power=0.0)
    clf.fit(x_tr[numeric], y_tr, x_va[numeric], y_va, ckpt_path=tmp_path / "ckpt.pth")
    assert clf.predict_proba(x_va[numeric]).shape == (45, 3)


def test_realmlp_per_sample_weight_length_mismatch_raises(tmp_path: Path) -> None:
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)

    clf = RealMLP_TD_Classifier(**_TINY)
    with pytest.raises(ValueError, match="per_sample_weight length"):
        clf.fit(
            x_tr,
            y_tr,
            x_va,
            y_va,
            ckpt_path=tmp_path / "ckpt.pth",
            per_sample_weight=np.ones(10),
        )


def test_realmlp_config_defaults_preserved() -> None:
    # The s6e6-proven defaults the port must not drift from.
    assert REALMLP_TD_CONFIG["epoch_select"] == "bacc"
    assert REALMLP_TD_CONFIG["loss_prior_power"] == 1.075
    assert REALMLP_TD_CONFIG["n_ens"] == 8
    assert REALMLP_TD_CONFIG["lr_sched"] == "flat_cos"
    assert REALMLP_TD_CONFIG["ema_decay"] == 0.99775
    # New training-dynamics knobs default OFF (stock path unchanged).
    assert REALMLP_TD_CONFIG["mixup_alpha"] == 0.0
    assert REALMLP_TD_CONFIG["sam_rho"] == 0.0
    assert REALMLP_TD_CONFIG["soft_ba_weight"] == 0.0


def test_realmlp_mixup_and_sam_paths(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)
    cats = ["cat_small", "cat_large"]

    mix = RealMLP_TD_Classifier(**_TINY, mixup_alpha=0.4)
    mix.fit(x_tr, y_tr, x_va, y_va, cat_col_names=cats, ckpt_path=tmp_path / "m.pth")
    proba = mix.predict_proba(x_va)
    assert proba.shape == (45, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    sam = RealMLP_TD_Classifier(**_TINY, sam_rho=0.05)
    sam.fit(x_tr, y_tr, x_va, y_va, cat_col_names=cats, ckpt_path=tmp_path / "s.pth")
    assert sam.predict_proba(x_va).shape == (45, 3)


def test_realmlp_soft_ba_paths(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)
    cats = ["cat_small", "cat_large"]

    # pure surrogate (replaces balanced softmax) and the 50/50 CE blend
    for w in (1.0, 0.5):
        clf = RealMLP_TD_Classifier(**_TINY, soft_ba_weight=w, loss_prior_power=0.0)
        clf.fit(
            x_tr, y_tr, x_va, y_va, cat_col_names=cats, ckpt_path=tmp_path / "b.pth"
        )
        proba = clf.predict_proba(x_va)
        assert proba.shape == (45, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_realmlp_mixup_and_sam_guards(tmp_path: Path) -> None:
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)

    with pytest.raises(ValueError, match="soft_ba_weight"):
        RealMLP_TD_Classifier(**_TINY, soft_ba_weight=0.5, mixup_alpha=0.2).fit(
            x_tr, y_tr, x_va, y_va, ckpt_path=tmp_path / "g.pth"
        )
    with pytest.raises(ValueError, match="allow_amp=False"):
        RealMLP_TD_Classifier(**_TINY, sam_rho=0.05, allow_amp=True).fit(
            x_tr, y_tr, x_va, y_va, ckpt_path=tmp_path / "g.pth"
        )
    with pytest.raises(ValueError, match="per_sample_weight"):
        RealMLP_TD_Classifier(**_TINY, mixup_alpha=0.2).fit(
            x_tr,
            y_tr,
            x_va,
            y_va,
            ckpt_path=tmp_path / "g.pth",
            per_sample_weight=np.ones(len(y_tr)),
        )


def _binary_frame(n: int, seed: int) -> "tuple[pd.DataFrame, NDArray[np.int64]]":
    """Two-class analogue of `_frame`, for the AUC epoch-selection path."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1], dtype=np.int64), n // 2 + 1)[:n]
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n) + y,
            "num_b": rng.normal(0, 1, n),
            "cat_small": rng.integers(0, 3, n),
            "cat_large": rng.integers(0, 15, n),
        }
    )
    return df, y


def test_realmlp_epoch_select_auc_binary(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _binary_frame(180, seed=0)
    x_va, y_va = _binary_frame(60, seed=1)
    cats = ["cat_small", "cat_large"]

    clf = RealMLP_TD_Classifier(**_TINY, epoch_select="auc", loss_prior_power=0.0)
    clf.fit(x_tr, y_tr, x_va, y_va, cat_col_names=cats, ckpt_path=tmp_path / "a.pth")

    # best_score_ is now an AUC, so it must live on the AUC scale.
    assert 0.0 <= clf.best_score_ <= 1.0
    assert clf.best_val_probs_ is not None
    assert clf.best_val_probs_.shape == (60, 2)
    proba = clf.predict_proba(x_va)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_realmlp_epoch_select_auc_rejects_multiclass(tmp_path: Path) -> None:
    x_tr, y_tr = _frame(120, seed=0)  # 3 classes
    x_va, y_va = _frame(45, seed=1)

    with pytest.raises(ValueError, match="binary-only"):
        RealMLP_TD_Classifier(**_TINY, epoch_select="auc").fit(
            x_tr, y_tr, x_va, y_va, ckpt_path=tmp_path / "a.pth"
        )


def test_realmlp_epoch_select_rejects_unknown(tmp_path: Path) -> None:
    x_tr, y_tr = _frame(120, seed=0)
    x_va, y_va = _frame(45, seed=1)

    # A typo must fail loudly rather than silently falling through to bacc.
    with pytest.raises(ValueError, match="epoch_select must be"):
        RealMLP_TD_Classifier(**_TINY, epoch_select="roc_auc").fit(
            x_tr, y_tr, x_va, y_va, ckpt_path=tmp_path / "a.pth"
        )


def test_realmlp_epoch_select_last_keeps_final_epoch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_tr, y_tr = _binary_frame(180, seed=0)
    x_va, y_va = _binary_frame(60, seed=1)
    params = {**_TINY, "epochs": 3, "use_early_stopping": True}
    clf = RealMLP_TD_Classifier(**params, epoch_select="last", loss_prior_power=0.0)
    clf.fit(x_tr, y_tr, x_va, y_va, ckpt_path=tmp_path / "a.pth")

    # The "score" is the epoch index, so the last epoch (index 2) is kept.
    assert clf.best_score_ == 2.0
    assert clf.best_val_probs_ is not None
    np.testing.assert_allclose(clf.predict_proba(x_va), clf.best_val_probs_, atol=1e-5)
