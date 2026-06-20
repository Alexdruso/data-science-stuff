"""Tests for data_science_stuff.kaggle_utils."""

from pathlib import Path

import numpy as np
import pandas as pd

from data_science_stuff.kaggle_utils import (
    classification_report_dict,
    print_classification_report,
    save_cv_result,
    tune_decision_weights,
    weighted_predict,
)


def _imbalanced_proba() -> tuple[np.ndarray, np.ndarray]:
    """3-class problem where argmax suppresses the rare class (recall 0).

    Class-2 rows put equal mass on class 0 and class 2, so argmax's first-max
    tie-break sends them all to class 0 — exactly the majority-prior failure
    mode that decision-weight tuning is meant to fix.
    """
    y = np.array([0] * 60 + [1] * 30 + [2] * 10, dtype=np.int64)
    proba = np.empty((len(y), 3), dtype=np.float64)
    proba[y == 0] = [0.70, 0.20, 0.10]
    proba[y == 1] = [0.20, 0.70, 0.10]
    proba[y == 2] = [0.45, 0.10, 0.45]
    return y, proba


def test_weighted_predict_basic() -> None:
    p = np.array([[0.6, 0.4], [0.55, 0.45]])
    assert weighted_predict(p, [1.0, 1.0]).tolist() == [0, 0]
    # Up-weighting class 1 flips both rows.
    assert weighted_predict(p, [1.0, 2.0]).tolist() == [1, 1]
    out = weighted_predict(p, [1.0, 1.0])
    assert out.shape == (2,)
    assert out.dtype == np.int64


def test_tune_decision_weights_recovers_rare_class() -> None:
    y, proba = _imbalanced_proba()

    argmax_acc = classification_report_dict(y, proba)["argmax_balanced_acc"]
    assert isinstance(argmax_acc, float)
    assert argmax_acc <= 0.7  # rare class has recall 0 under argmax

    weights = tune_decision_weights(y, proba, n_restarts=4)
    assert weights.shape == (3,)
    assert np.isclose(weights.max(), 1.0)  # normalized so max weight is 1
    assert weights[2] > weights[0] and weights[2] > weights[1]  # rare class up-weighted

    tuned_acc = classification_report_dict(y, proba, weights=weights)[
        "weighted_balanced_acc"
    ]
    assert isinstance(tuned_acc, float)
    assert tuned_acc > argmax_acc
    assert tuned_acc >= 0.99  # all three classes recoverable here


def test_classification_report_dict_keys() -> None:
    y, proba = _imbalanced_proba()

    plain = classification_report_dict(y, proba, labels=["GALAXY", "QSO", "STAR"])
    assert set(plain) == {
        "argmax_balanced_acc",
        "per_class_recall_argmax",
        "confusion_argmax",
    }
    recall = plain["per_class_recall_argmax"]
    assert isinstance(recall, dict)
    assert set(recall) == {"GALAXY", "QSO", "STAR"}
    assert recall["STAR"] == 0.0  # rare class suppressed by argmax

    weighted = classification_report_dict(y, proba, weights=[1.0, 1.0, 2.0])
    assert "weighted_balanced_acc" in weighted
    assert "per_class_recall_weighted" in weighted
    assert "confusion_weighted" in weighted


def test_print_classification_report_smoke(capsys) -> None:  # type: ignore[no-untyped-def]
    y, proba = _imbalanced_proba()
    report = classification_report_dict(y, proba, weights=[1.0, 1.0, 2.0])
    print_classification_report(report)
    captured = capsys.readouterr().out
    assert "argmax   balanced_acc" in captured
    assert "weighted balanced_acc" in captured

    # Without weights, only the argmax block prints.
    print_classification_report(classification_report_dict(y, proba))
    out2 = capsys.readouterr().out
    assert "weighted balanced_acc" not in out2


def test_save_cv_result_creates_and_appends(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    save_cv_result(results_dir, "run_a", [0.9, 0.91], 0.905, metric_name="bal_acc")

    path = results_dir / "cv_scores.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert df.loc[0, "model"] == "run_a"
    assert df.loc[0, "oof_bal_acc"] == 0.905
    assert df.loc[0, "fold_1_bal_acc"] == 0.9

    save_cv_result(results_dir, "run_b", [0.8], 0.8, metric_name="bal_acc")
    df2 = pd.read_csv(path)
    assert len(df2) == 2
    assert df2.loc[1, "model"] == "run_b"
