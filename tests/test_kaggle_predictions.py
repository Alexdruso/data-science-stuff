"""Tests for keyed prediction files and the shared fold splitter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from data_science_stuff.kaggle.cv import fold_assignment, make_splitter, run_cv
from data_science_stuff.kaggle.predictions import (
    list_predictions,
    load_folds,
    load_predictions,
    migrate_npy,
    save_predictions,
)


def _toy() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    rng = np.random.default_rng(0)
    train_ids = rng.permutation(np.arange(100, 140))
    test_ids = np.arange(500, 510)
    oof = rng.random(len(train_ids))
    test = rng.random(len(test_ids))
    return train_ids, test_ids, oof, test


def test_round_trip_reorders_by_id(tmp_path: Path) -> None:
    train_ids, test_ids, oof, test = _toy()
    folds = np.arange(len(train_ids)) % 5 + 1
    paths = save_predictions(
        tmp_path,
        "m",
        oof=oof,
        test=test,
        train_ids=train_ids,
        test_ids=test_ids,
        folds=folds,
    )
    assert paths.oof.name == "oof_m.parquet"

    order = np.sort(train_ids)  # a reader using a different row order
    got = load_predictions(tmp_path, "m", order)
    expected = oof[np.argsort(train_ids)]
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(
        load_predictions(tmp_path, "m", test_ids, split="test"), test
    )
    np.testing.assert_array_equal(
        load_folds(tmp_path, "m", order), folds[np.argsort(train_ids)]
    )


def test_alignment_errors(tmp_path: Path) -> None:
    train_ids, test_ids, oof, test = _toy()
    save_predictions(
        tmp_path, "m", oof=oof, test=test, train_ids=train_ids, test_ids=test_ids
    )
    with pytest.raises(ValueError, match="not in the file"):
        load_predictions(tmp_path, "m", np.r_[train_ids[:-1], 9999])
    with pytest.raises(ValueError, match="allow_subset"):
        load_predictions(tmp_path, "m", train_ids[:10])
    sub = load_predictions(tmp_path, "m", train_ids[:10], allow_subset=True)
    np.testing.assert_allclose(sub, oof[:10])
    with pytest.raises(FileNotFoundError, match="fold"):
        load_folds(tmp_path, "m", train_ids)
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path, "absent", train_ids)


def test_save_validation(tmp_path: Path) -> None:
    train_ids, test_ids, oof, test = _toy()
    with pytest.raises(ValueError, match="ids for"):
        save_predictions(
            tmp_path,
            "m",
            oof=oof,
            test=test,
            train_ids=train_ids[:-1],
            test_ids=test_ids,
        )
    dup = train_ids.copy()
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="not unique"):
        save_predictions(
            tmp_path, "m", oof=oof, test=test, train_ids=dup, test_ids=test_ids
        )
    with pytest.raises(ValueError, match="disagree"):
        save_predictions(
            tmp_path,
            "m",
            oof=np.c_[oof, oof],
            test=test,
            train_ids=train_ids,
            test_ids=test_ids,
        )
    with pytest.raises(ValueError, match="fold labels"):
        save_predictions(
            tmp_path,
            "m",
            oof=oof,
            test=test,
            train_ids=train_ids,
            test_ids=test_ids,
            folds=np.array([1]),
        )
    with pytest.raises(ValueError, match="1-D or 2-D"):
        save_predictions(
            tmp_path,
            "m",
            oof=oof.reshape(-1, 1, 1),
            test=test.reshape(-1, 1, 1),
            train_ids=train_ids,
            test_ids=test_ids,
        )


def test_multiclass_and_legacy_npy(tmp_path: Path) -> None:
    train_ids, test_ids, _, _ = _toy()
    oof = np.random.default_rng(1).random((len(train_ids), 3))
    test = np.random.default_rng(2).random((len(test_ids), 3))
    save_predictions(
        tmp_path,
        "mc",
        oof=oof,
        test=test,
        train_ids=train_ids,
        test_ids=test_ids,
        also_npy=True,
    )
    np.testing.assert_allclose(load_predictions(tmp_path, "mc", train_ids), oof)
    assert (tmp_path / "oof_mc.npy").exists()

    np.save(tmp_path / "oof_old.npy", oof[:, 0])
    np.save(tmp_path / "test_old.npy", test[:, 0])
    with pytest.warns(UserWarning, match="positional legacy"):
        got = load_predictions(tmp_path, "old", train_ids)
    np.testing.assert_allclose(got, oof[:, 0])
    with pytest.raises(ValueError, match="rows but"):
        load_predictions(tmp_path, "old", train_ids[:3])
    (tmp_path / "oof_lonely.npy").touch()
    (tmp_path / "oof_notes.txt").touch()
    assert list_predictions(tmp_path) == ["mc", "old"]

    paths = migrate_npy(tmp_path, "old", train_ids=train_ids, test_ids=test_ids)
    assert paths.oof.exists()
    np.testing.assert_allclose(load_predictions(tmp_path, "old", train_ids), oof[:, 0])


def test_fold_assignment_matches_run_cv() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 60)
    x = rng.random((60, 2))

    def fit_fold(*args):
        # Predict the fold number, so the OOF array IS run_cv's fold assignment.
        x_va, x_te, fold = args[2], args[4], args[5]
        return np.full(len(x_va), float(fold)), np.zeros(len(x_te))

    res = run_cv(x, y, x[:5], fit_fold, n_splits=4)
    np.testing.assert_array_equal(fold_assignment(y, n_splits=4), res.oof.astype(int))
    assert make_splitter(stratified=False).__class__.__name__ == "KFold"
