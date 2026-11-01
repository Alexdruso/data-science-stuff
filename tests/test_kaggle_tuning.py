"""Tests for data_science_stuff.kaggle.tuning."""

import json
from pathlib import Path

import optuna
import pytest

from data_science_stuff.kaggle.io import load_params
from data_science_stuff.kaggle.tuning import run_study

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -2.0, 2.0)
    return -((x - 1.0) ** 2)


def test_run_study_writes_best_params_readable_by_load_params(tmp_path: Path) -> None:
    out = tmp_path / "best_params_toy.json"
    study = run_study(
        _objective, n_trials=15, params_path=out, fixed={"n_estimators": 100}
    )
    written = json.loads(out.read_text())
    assert written == {"n_estimators": 100, "x": study.best_params["x"]}
    merged = load_params(tmp_path, {"x": 0.0, "lr": 0.1}, "best_params_toy.json")
    assert merged["lr"] == 0.1
    assert merged["x"] == study.best_params["x"]


def test_enqueued_defaults_run_first() -> None:
    study = run_study(_objective, n_trials=3, enqueue=[{"x": 1.0}])
    assert study.trials[0].params == {"x": 1.0}
    assert study.best_value == 0.0


def test_study_resumes_from_storage_without_re_enqueueing(tmp_path: Path) -> None:
    db = tmp_path / "study.db"
    kwargs = {"study_name": "toy", "storage_path": db, "enqueue": [{"x": 0.5}]}
    run_study(_objective, n_trials=2, **kwargs)  # type: ignore[arg-type]
    study = run_study(_objective, n_trials=2, **kwargs)  # type: ignore[arg-type]
    assert len(study.trials) == 4
    assert sum(t.params == {"x": 0.5} for t in study.trials) == 1


def test_storage_without_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="study_name"):
        run_study(_objective, n_trials=1, storage_path=tmp_path / "s.db")


def test_minimize_direction() -> None:
    study = run_study(
        lambda t: (t.suggest_float("x", -1, 1)) ** 2,
        n_trials=10,
        direction="minimize",
        cuda_cleanup=False,
    )
    assert study.best_value < 0.2
