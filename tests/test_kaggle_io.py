"""Tests for data_science_stuff.kaggle.io."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_science_stuff.kaggle.io import (
    competition_dirs,
    load_params,
    save_threshold_weights,
    write_submission,
)


def test_competition_dirs_layout(tmp_path: Path) -> None:
    script = tmp_path / "playground-series-s7e1" / "src" / "train_lgbm.py"
    dirs = competition_dirs(script)
    root = tmp_path / "playground-series-s7e1"
    assert dirs.data == root / "data"
    assert dirs.results == root / "results"
    assert dirs.submissions == root / "submissions"
    # Unpacks in the same order as the old per-script path trio.
    data_dir, results_dir, submissions_dir = dirs
    assert (data_dir, results_dir, submissions_dir) == tuple(dirs)


def test_load_params_returns_defaults_copy_when_no_file(tmp_path: Path) -> None:
    defaults = {"n_estimators": 100, "learning_rate": 0.05}
    params = load_params(tmp_path, defaults, "best_params.json")
    assert params == defaults
    assert params is not defaults


def test_load_params_merges_tuned_file_without_mutating_defaults(
    tmp_path: Path,
) -> None:
    defaults = {"n_estimators": 100, "learning_rate": 0.05}
    tuned = {"learning_rate": 0.01, "num_leaves": 63}
    (tmp_path / "best_params_lgbm.json").write_text(json.dumps(tuned))

    params = load_params(tmp_path, defaults, "best_params_lgbm.json")
    assert params == {"n_estimators": 100, "learning_rate": 0.01, "num_leaves": 63}
    assert defaults == {"n_estimators": 100, "learning_rate": 0.05}


def test_write_submission_round_trip(tmp_path: Path) -> None:
    ids = np.array([11, 7, 3])
    preds = np.array(["GALAXY", "STAR", "QSO"])
    out = write_submission(tmp_path / "submissions", "lgbm_v1", ids, "class", preds)
    assert out == tmp_path / "submissions" / "lgbm_v1.csv"

    frame = pd.read_csv(out)
    assert list(frame.columns) == ["id", "class"]
    assert frame["id"].tolist() == [11, 7, 3]
    assert frame["class"].tolist() == ["GALAXY", "STAR", "QSO"]


def test_write_submission_keeps_explicit_csv_suffix_and_custom_id(
    tmp_path: Path,
) -> None:
    out = write_submission(
        tmp_path, "sub.csv", np.array([1]), "y", np.array([0.5]), id_col="row_id"
    )
    assert out.name == "sub.csv"
    assert list(pd.read_csv(out).columns) == ["row_id", "y"]


def test_save_threshold_weights_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "threshold_weights_lgbm.json"
    returned = save_threshold_weights(
        np.array([1.0, 2.5, 0.75]), ["GALAXY", "QSO", "STAR"], path
    )
    assert returned == path
    with path.open() as f:
        assert json.load(f) == {"GALAXY": 1.0, "QSO": 2.5, "STAR": 0.75}
