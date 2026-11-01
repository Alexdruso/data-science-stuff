"""Tests for helpers promoted from competition code into existing modules.

Covers ``cv.run_cv_seeds``, the XGBoost/CatBoost helpers in ``device``,
``stacking.ridge_calibrate``/``load_public_predictions``,
``decision.xgb_balanced_error_metric``, ``encoding.rank_gauss``/
``fourier_features``, ``shift.uniform_remask`` and ``fit_mlp_fold``'s
``epoch_select``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import xgboost
from numpy.typing import NDArray

from data_science_stuff.kaggle import device
from data_science_stuff.kaggle.cv import Matrix, run_cv, run_cv_seeds
from data_science_stuff.kaggle.decision import xgb_balanced_error_metric
from data_science_stuff.kaggle.encoding import fourier_features, rank_gauss
from data_science_stuff.kaggle.models.mlp import fit_mlp_fold
from data_science_stuff.kaggle.shift import uniform_remask
from data_science_stuff.kaggle.stacking import (
    load_public_predictions,
    ridge_calibrate,
)

# --- cv.run_cv_seeds -------------------------------------------------------


def _make_mean_fit(seed: int) -> Any:
    def fit_fold(
        _x_tr: Matrix,
        y_tr: "NDArray[np.int64]",
        x_va: Matrix,
        _y_va: "NDArray[np.int64]",
        x_te: Matrix,
        fold: int,
    ) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
        level = y_tr.mean() + seed + fold / 100
        return np.full(len(x_va), level), np.full(len(x_te), level)

    return fit_fold


def test_run_cv_seeds_averages_per_seed_run_cv() -> None:
    x = np.arange(60, dtype=float).reshape(30, 2)
    y = np.array([0, 1, 1] * 10)
    res = run_cv_seeds(x, y, x[:5], _make_mean_fit, seeds=(1, 2), n_splits=3)
    singles = [
        run_cv(x, y, x[:5], _make_mean_fit(s), n_splits=3, seed=s) for s in (1, 2)
    ]
    np.testing.assert_allclose(res.per_seed_oof[0], singles[0].oof)
    np.testing.assert_allclose(res.oof, (singles[0].oof + singles[1].oof) / 2)
    np.testing.assert_allclose(res.test, (singles[0].test + singles[1].test) / 2)
    assert res.per_seed_test.shape == (2, 5)


def _val_rows_per_seed(*, resplit: bool) -> "dict[int, list[list[int]]]":
    """Run run_cv_seeds on x = row index and record each fold's validation rows."""
    x = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 15)
    seen: dict[int, list[list[int]]] = {}

    def make(seed: int) -> Any:
        def fit_fold(*args: Any) -> Any:
            x_va, x_te = args[2], args[4]
            seen.setdefault(seed, []).append(sorted(x_va[:, 0].astype(int)))
            return np.zeros(len(x_va)), np.zeros(len(x_te))

        return fit_fold

    res = run_cv_seeds(
        x,
        y,
        x[:2],
        make,
        seeds=(1, 2),
        resplit=resplit,
        n_splits=3,
        score_fn=lambda *_: 0.0,
    )
    assert len(res.fold_scores) == 6
    return seen


def test_run_cv_seeds_fixed_split_shares_folds() -> None:
    fixed = _val_rows_per_seed(resplit=False)
    assert fixed[1] == fixed[2]
    resplit = _val_rows_per_seed(resplit=True)
    assert resplit[1] != resplit[2]


def test_run_cv_seeds_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_cv_seeds(
            np.zeros((4, 1)),
            np.array([0, 1, 0, 1]),
            np.zeros((1, 1)),
            _make_mean_fit,
            seeds=(),
        )


# --- device: XGBoost / CatBoost -------------------------------------------


@pytest.fixture
def _fresh_xgb_probe() -> Iterator[None]:
    device.get_xgb_device.cache_clear()
    yield
    device.get_xgb_device.cache_clear()


@pytest.mark.usefixtures("_fresh_xgb_probe")
def test_xgb_device_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XGB_DEVICE", "cpu")
    assert device.xgb_device_params() == {"tree_method": "hist", "device": "cpu"}


@pytest.mark.usefixtures("_fresh_xgb_probe")
def test_xgb_device_cpu_without_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XGB_DEVICE", raising=False)
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})
    assert device.get_xgb_device() == "cpu"


@pytest.mark.usefixtures("_fresh_xgb_probe")
def test_xgb_device_follows_torch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XGB_DEVICE", raising=False)
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.get_xgb_device() == "cuda"
    device.get_xgb_device.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.get_xgb_device() == "cpu"


def test_catboost_gpu_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CB_TASK_TYPE", raising=False)
    monkeypatch.delenv("CB_GPU_RAM_PART", raising=False)
    assert device.catboost_gpu_params() == {
        "task_type": "GPU",
        "border_count": 254,
        "max_ctr_complexity": 1,
        "gpu_ram_part": 0.85,
    }
    monkeypatch.setenv("CB_GPU_RAM_PART", "0.35")
    assert device.catboost_gpu_params()["gpu_ram_part"] == 0.35
    assert device.catboost_gpu_params(gpu_ram_part=0.5)["gpu_ram_part"] == 0.5
    monkeypatch.setenv("CB_TASK_TYPE", "CPU")
    assert device.catboost_gpu_params() == {"task_type": "CPU", "border_count": 254}


# --- stacking --------------------------------------------------------------


def test_ridge_calibrate_shapes_and_fit_on_train_only() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    b1 = np.eye(3)[y] * 2 + rng.normal(0, 0.5, (200, 3))
    b2 = rng.normal(0, 1, (200, 3))
    tr, va = np.arange(150), np.arange(150, 200)
    x_tr, x_va = ridge_calibrate([b1[tr], b2[tr]], [[b1[va], b2[va]]], y[tr])
    assert x_tr.shape == (150, 6)
    assert x_va.shape == (50, 6)
    # the informative base maps close to one-hot; argmax recovers labels
    assert (x_va[:, :3].argmax(axis=1) == y[va]).mean() > 0.8


def test_ridge_calibrate_accepts_1d_blocks_and_checks_eval_sets() -> None:
    y = np.array([0, 1] * 20)
    p = y + np.linspace(0, 0.1, 40)
    (x_tr,) = ridge_calibrate([p], [], y)
    assert x_tr.shape == (40, 2)
    with pytest.raises(ValueError, match="one block per base"):
        ridge_calibrate([p, p], [[p]], y)


def test_load_public_predictions_validates_pairs(tmp_path: Path) -> None:
    good, bad = tmp_path / "nb_good", tmp_path / "nb_bad"
    good.mkdir()
    bad.mkdir()
    np.save(good / "oof_lgbm.npy", np.linspace(0, 1, 10))
    np.save(good / "test_lgbm.npy", np.linspace(0, 1, 4))
    np.save(bad / "oof_short.npy", np.zeros(9))
    np.save(bad / "test_short.npy", np.zeros(4))
    np.save(bad / "oof_nan.npy", np.full(10, np.nan))
    np.save(bad / "test_nan.npy", np.zeros(4))
    np.save(bad / "oof_orphan.npy", np.zeros(10))
    np.save(bad / "oof_wide.npy", np.zeros((10, 3)))
    np.save(bad / "test_wide.npy", np.zeros((4, 2)))

    res = load_public_predictions([good, bad], n_train=10, n_test=4)
    assert res.names == ["nb_good/lgbm"]
    assert res.oof[0].shape == (10,)
    assert set(res.skipped) == {
        "nb_bad/short",
        "nb_bad/nan",
        "nb_bad/orphan",
        "nb_bad/wide",
    }


# --- decision --------------------------------------------------------------


def test_xgb_balanced_error_metric() -> None:
    metric = xgb_balanced_error_metric(3)
    y = np.array([0, 1, 2, 2])
    perfect = np.eye(3)[y]
    assert metric(y, perfect) == 0.0
    assert metric(y, perfect.ravel(), sample_weight=np.ones(4)) == 0.0
    all_zero = np.tile([1.0, 0.0, 0.0], (4, 1))
    assert metric(y, all_zero) == pytest.approx(2 / 3)


# --- encoding --------------------------------------------------------------


def test_rank_gauss_is_monotone_and_centred() -> None:
    v = np.array([5.0, 1.0, np.nan, 100.0, 3.0])
    out = rank_gauss(v)
    assert out.dtype == np.float32
    finite = ~np.isnan(v)
    assert (np.argsort(out[finite]) == np.argsort(v[finite])).all()
    assert abs(float(out.mean())) < 0.3
    assert (rank_gauss(np.full(3, np.nan)) == 0).all()


def test_fourier_features_values_and_order() -> None:
    x = np.array([[0.25, 0.5]])
    out = fourier_features(x, 2)
    expected_sin = np.sin(np.pi * np.array([0.25, 0.5, 0.5, 1.0]))
    expected_cos = np.cos(np.pi * np.array([0.25, 0.5, 0.5, 1.0]))
    np.testing.assert_allclose(out[0], np.concatenate([expected_sin, expected_cos]))
    assert fourier_features(x, 0).shape == (1, 0)


# --- shift -----------------------------------------------------------------


def test_uniform_remask_matches_test_rate_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=20000), "b": rng.normal(size=20000)})
    test = pd.DataFrame({"a": [np.nan] * 3 + [0.0] * 7, "b": [0.0] * 10})
    out = uniform_remask(train, test, ["a", "missing_col"])
    assert out["a"].isna().mean() == pytest.approx(0.3, abs=0.02)
    assert out["b"].notna().all()
    assert train["a"].notna().all()  # input untouched
    pd.testing.assert_frame_equal(out, uniform_remask(train, test, ["a"]))
    half = uniform_remask(train, test, ["a"], mult=0.5)
    assert half["a"].isna().mean() == pytest.approx(0.15, abs=0.02)


# --- models.mlp epoch_select ----------------------------------------------

_PARAMS: "dict[str, object]" = {
    "n_layers": 1,
    "first_size": 8,
    "dropout": 0.0,
    "lr": 1e-2,
    "weight_decay": 0.0,
}


def test_fit_mlp_fold_epoch_select_last_ignores_score() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    calls: list[int] = []

    def score(_y: Any, _p: Any) -> float:
        calls.append(1)
        return 0.0

    val, test = fit_mlp_fold(
        x[:80],
        y[:80],
        x[80:],
        y[80:],
        x[:5],
        num_idx=[0, 1, 2],
        cat_idx=[],
        params=_PARAMS,
        n_classes=2,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=32,
        patience=1,
        score_fn=score,
        epoch_select="last",
    )
    assert calls == []
    assert val.shape == (40, 2)
    assert test.shape == (5, 2)
    np.testing.assert_allclose(val.sum(axis=1), 1.0, atol=1e-5)


def test_fit_mlp_fold_epoch_select_validates() -> None:
    x = np.zeros((4, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="epoch_select"):
        fit_mlp_fold(
            x,
            y,
            x,
            y,
            x,
            num_idx=[0],
            cat_idx=[],
            params=_PARAMS,
            n_classes=2,
            epoch_select="middle",  # type: ignore[arg-type]
        )
