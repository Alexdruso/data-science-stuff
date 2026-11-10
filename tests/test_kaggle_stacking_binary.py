"""Tests for the binary pool-stacking helpers in data_science_stuff.kaggle.stacking."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data_science_stuff.kaggle.stacking import (
    load_oof_banks,
    logit,
    nested_lr_stack,
    rank_logit,
)


def test_logit_inverts_sigmoid_and_clips() -> None:
    x = np.array([-3.0, 0.0, 2.5])
    np.testing.assert_allclose(logit(1 / (1 + np.exp(-x))), x, atol=1e-12)
    out = logit(np.array([0.0, 1.0]), eps=1e-7)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [-out[1], out[1]])


def test_rank_logit_is_monotone_symmetric_and_scale_free() -> None:
    v = np.random.default_rng(0).normal(size=101)
    r = rank_logit(v)
    assert (np.argsort(r) == np.argsort(v)).all()
    np.testing.assert_allclose(np.sort(r), -np.sort(r)[::-1], atol=1e-12)
    np.testing.assert_array_equal(rank_logit(np.exp(v)), r)


def _toy(n: int = 600) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(1)
    y = (rng.random(n) < 0.3).astype(int)
    x = np.column_stack([y + rng.normal(scale=s, size=n) for s in (0.8, 1.2, 3.0)])
    return x, y


def test_nested_lr_stack_matches_reference_loop() -> None:
    x, y = _toy()
    xt = x[:50] + 0.1
    res = nested_lr_stack(x, y, xt, c=0.1, seeds=(7, 13), cols=[0, 2])
    oof, tst = np.zeros(len(y)), np.zeros(len(xt))
    for s in (7, 13):
        for trn, va in StratifiedKFold(5, shuffle=True, random_state=s).split(x, y):
            m = LogisticRegression(C=0.1, max_iter=5000).fit(x[trn][:, [0, 2]], y[trn])
            oof[va] += m.decision_function(x[va][:, [0, 2]]) / 2
            tst += m.decision_function(xt[:, [0, 2]]) / 10
    np.testing.assert_array_equal(res.oof, oof)
    assert res.test is not None
    np.testing.assert_allclose(res.test, tst)
    assert 0.7 < res.auc <= 1.0


def test_nested_lr_stack_without_test() -> None:
    x, y = _toy()
    auc, oof, tst = nested_lr_stack(x, y, n_splits=3, seeds=(0,))
    assert tst is None
    assert oof.shape == y.shape
    assert auc > 0.7


def test_load_oof_banks_reads_every_layout(tmp_path: Path) -> None:
    n_tr, n_te = 6, 4
    rng = np.random.default_rng(2)
    a = tmp_path / "srcA"
    a.mkdir()
    np.save(a / "gb_oof.npy", rng.random(n_tr))
    np.save(a / "gb_full.npy", rng.random(n_te))
    np.save(a / "oof_xgb.npy", rng.random((n_tr, 2)))
    np.save(a / "refit_xgb.npy", rng.random((n_te, 2)))
    np.save(a / "oof_bad.npy", rng.random(n_tr + 1))  # wrong rows -> skipped
    np.save(a / "test_bad.npy", rng.random(n_te))
    np.save(a / "oof_nan.npy", np.full(n_tr, np.nan))  # non-finite -> skipped
    np.save(a / "test_nan.npy", rng.random(n_te))
    np.save(a / "oof_orphan.npy", rng.random(n_tr))  # no test file -> skipped
    lm = tmp_path / "srcB" / "masa_lm"
    lm.mkdir(parents=True)
    np.save(lm / "oof.npy", rng.random(n_tr))
    np.save(lm / "test.npy", rng.random(n_te))

    banks = load_oof_banks([a, tmp_path / "srcB"], n_train=n_tr, n_test=n_te)

    assert sorted(banks) == ["masa_lm", "srcA:gb", "srcA:xgb"]
    o, t = banks["srcA:xgb"]
    np.testing.assert_array_equal(o, np.load(a / "oof_xgb.npy")[:, 1])
    np.testing.assert_array_equal(t, np.load(a / "refit_xgb.npy")[:, 1])
    assert banks["masa_lm"][0].shape == (n_tr,)
