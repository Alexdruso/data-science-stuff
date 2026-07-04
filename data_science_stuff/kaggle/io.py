"""Competition I/O helpers: standard paths, tuned params, submissions.

Every competition in this repo follows the same layout::

    playground-series-<id>/
    ├── data/         (gitignored CSVs)
    ├── results/      (oof_*.npy, test_*.npy, cv_scores.csv, best_params_*.json)
    ├── submissions/
    └── src/          (features.py, train_<model>.py, ...)

These helpers replace the path/params/submission boilerplate previously
copy-pasted into every training script.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class CompetitionDirs(NamedTuple):
    """The three standard per-competition directories."""

    data: Path
    results: Path
    submissions: Path


def competition_dirs(script_file: str | Path) -> CompetitionDirs:
    """Resolve the standard ``data``/``results``/``submissions`` directories.

    Assumes the calling script lives in ``<competition>/src/`` with the three
    directories as siblings of ``src/``. Call as ``competition_dirs(__file__)``.

    Args:
        script_file: Path of the calling script (typically ``__file__``).

    Returns:
        Named tuple ``(data, results, submissions)``. Directories are not
        created; ``results``/``submissions`` are created lazily by the writers.
    """
    root = Path(script_file).resolve().parent.parent
    return CompetitionDirs(root / "data", root / "results", root / "submissions")


def load_params(
    results_dir: Path,
    defaults: Mapping[str, object],
    filename: str = "best_params.json",
) -> dict[str, object]:
    """Merge tuned params from ``results_dir/filename`` over ``defaults``.

    The tuned file is typically written by an Optuna ``tune_<model>.py`` run;
    when it is absent the defaults are returned unchanged, so training scripts
    work before any tuning has happened.

    Args:
        results_dir: The competition's ``results/`` directory.
        defaults: Baseline parameter dict (never mutated).
        filename: JSON file with the tuned overrides.

    Returns:
        A new dict: ``defaults`` updated with the JSON contents if present.
    """
    params: dict[str, object] = dict(defaults)
    path = results_dir / filename
    if path.exists():
        with path.open() as f:
            params.update(json.load(f))
    return params


def write_submission(
    submissions_dir: Path,
    name: str,
    ids: NDArray[Any],
    target: str,
    preds: NDArray[Any],
    *,
    id_col: str = "id",
) -> Path:
    """Write a ``(id_col, target)`` submission CSV and return its path.

    ``ids`` must come from ``build_features()`` output (row-order invariant):
    the prediction arrays are stored in ``build_features()`` sort order, so
    ids loaded any other way silently misalign.

    Args:
        submissions_dir: The competition's ``submissions/`` directory
            (created if absent).
        name: Submission name; ``.csv`` is appended when missing.
        ids: Row identifiers, aligned with ``preds``.
        target: Target column name.
        preds: Predictions (probabilities or labels).
        id_col: Name of the identifier column.

    Returns:
        Path of the written CSV.
    """
    submissions_dir.mkdir(exist_ok=True)
    filename = name if name.endswith(".csv") else f"{name}.csv"
    out_path = submissions_dir / filename
    pd.DataFrame({id_col: ids, target: preds}).to_csv(out_path, index=False)
    return out_path


def save_threshold_weights(
    weights: NDArray[np.float64],
    class_names: Sequence[str],
    path: Path,
) -> Path:
    """Dump per-class decision weights as ``{class_name: weight}`` JSON.

    Args:
        weights: Per-class multiplicative weights (see
            :func:`data_science_stuff.kaggle.decision.optimize_thresholds`).
        class_names: Class names in encoded order (``LabelEncoder.classes_``).
        path: Destination JSON path, e.g.
            ``results/threshold_weights_<run>.json``.

    Returns:
        The written path.
    """
    with path.open("w") as f:
        json.dump(dict(zip(class_names, np.asarray(weights).tolist())), f, indent=2)
    return path
