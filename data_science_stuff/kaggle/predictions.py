"""Keyed OOF/test predictions: parquet files that carry their own row ids.

Positional ``oof_<name>.npy`` arrays are only correct if every reader loads
``y`` and ids in exactly the order the writer used (the row-order invariant;
breaking it once produced a 0.5-AUC submission). Here each prediction file
stores the ``id`` and, for OOF, the validation ``fold`` of every row:

* ``results/oof_<name>.parquet`` — ``id``, ``fold``, ``pred`` (or ``pred_0..k``)
* ``results/test_<name>.parquet`` — ``id``, ``pred`` (or ``pred_0..k``)

:func:`load_predictions` re-orders by id to whatever order the caller asks
for and fails loudly on missing, extra or duplicate ids, so misalignment
becomes an exception instead of a silently wrong score. Legacy ``.npy``
pairs still load (with a warning) so older competitions keep working.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Split = Literal["oof", "test"]


class PredictionPaths(NamedTuple):
    """Paths written by :func:`save_predictions`."""

    oof: Path
    test: Path


def _pred_columns(preds: NDArray[Any]) -> dict[str, NDArray[Any]]:
    if preds.ndim == 1:
        return {"pred": preds}
    if preds.ndim == 2:  # noqa: PLR2004 — (rows, classes)
        return {f"pred_{j}": preds[:, j] for j in range(preds.shape[1])}
    msg = f"predictions must be 1-D or 2-D, got shape {preds.shape}"
    raise ValueError(msg)


def _check_ids(ids: NDArray[Any], n: int, what: str) -> None:
    if len(ids) != n:
        msg = f"{what}: {len(ids)} ids for {n} prediction rows"
        raise ValueError(msg)
    if pd.Index(ids).has_duplicates:
        msg = f"{what}: ids are not unique"
        raise ValueError(msg)


def save_predictions(
    results_dir: Path,
    name: str,
    *,
    oof: NDArray[Any],
    test: NDArray[Any],
    train_ids: NDArray[Any],
    test_ids: NDArray[Any],
    folds: NDArray[Any] | None = None,
    id_col: str = "id",
    also_npy: bool = False,
) -> PredictionPaths:
    """Write ``oof_<name>.parquet`` and ``test_<name>.parquet``.

    Args:
        results_dir: The competition's ``results/`` directory (created if absent).
        name: Run name (file stem). Use the same name as the registry run.
        oof: Out-of-fold predictions, row ``i`` belonging to ``train_ids[i]``.
        test: Test predictions, row ``i`` belonging to ``test_ids[i]``.
        train_ids / test_ids: Row ids aligned with ``oof`` / ``test``.
        folds: Validation fold per training row (``cv.fold_assignment``).
            Strongly recommended: it lets readers verify fold alignment
            before stacking two runs.
        id_col: Name of the id column.
        also_npy: Also write legacy positional ``.npy`` files for old
            readers that glob ``oof_*.npy``.

    Returns:
        :class:`PredictionPaths`.
    """
    oof_arr, test_arr = np.asarray(oof), np.asarray(test)
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    _check_ids(train_ids, len(oof_arr), "oof")
    _check_ids(test_ids, len(test_arr), "test")
    if oof_arr.shape[1:] != test_arr.shape[1:]:
        msg = f"oof shape {oof_arr.shape} and test shape {test_arr.shape} disagree"
        raise ValueError(msg)

    results_dir.mkdir(parents=True, exist_ok=True)
    oof_frame: dict[str, Any] = {id_col: train_ids}
    if folds is not None:
        fold_arr = np.asarray(folds)
        if len(fold_arr) != len(oof_arr):
            msg = f"{len(fold_arr)} fold labels for {len(oof_arr)} oof rows"
            raise ValueError(msg)
        oof_frame["fold"] = fold_arr.astype(np.int16)
    oof_frame.update(_pred_columns(oof_arr))
    paths = PredictionPaths(
        results_dir / f"oof_{name}.parquet", results_dir / f"test_{name}.parquet"
    )
    pd.DataFrame(oof_frame).to_parquet(paths.oof, index=False)
    pd.DataFrame({id_col: test_ids, **_pred_columns(test_arr)}).to_parquet(
        paths.test, index=False
    )
    if also_npy:
        np.save(results_dir / f"oof_{name}.npy", oof_arr)
        np.save(results_dir / f"test_{name}.npy", test_arr)
    return paths


def _read(results_dir: Path, name: str, split: Split) -> pd.DataFrame | None:
    path = results_dir / f"{split}_{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _aligned(
    frame: pd.DataFrame,
    ids: NDArray[Any],
    id_col: str,
    column_prefix: str,
    where: str,
    *,
    allow_subset: bool,
) -> pd.DataFrame:
    if id_col not in frame:
        msg = f"{where}: no {id_col!r} column"
        raise ValueError(msg)
    index = pd.Index(frame[id_col])
    if index.has_duplicates:
        msg = f"{where}: duplicate ids in file"
        raise ValueError(msg)
    pos = index.get_indexer(pd.Index(ids))
    missing = int((pos < 0).sum())
    if missing:
        msg = f"{where}: {missing} of {len(ids)} requested ids are not in the file"
        raise ValueError(msg)
    if not allow_subset and len(index) != len(ids):
        msg = (
            f"{where}: file has {len(index)} rows but {len(ids)} ids were requested "
            "(pass allow_subset=True to select a subset)"
        )
        raise ValueError(msg)
    cols = [
        c
        for c in frame.columns
        if c == column_prefix or c.startswith(f"{column_prefix}_")
    ]
    selected: pd.DataFrame = frame.iloc[pos][cols]
    return selected


def load_predictions(
    results_dir: Path,
    name: str,
    ids: NDArray[Any],
    *,
    split: Split = "oof",
    id_col: str = "id",
    allow_subset: bool = False,
) -> NDArray[np.float64]:
    """Load predictions re-ordered to ``ids``.

    Args:
        results_dir: The competition's ``results/`` directory.
        name: Run name (``oof_<name>`` / ``test_<name>``).
        ids: The row order you need — e.g. ids from ``build_features()``.
        split: ``"oof"`` or ``"test"``.
        id_col: Name of the id column.
        allow_subset: Allow ``ids`` to be a strict subset of the file's rows.

    Returns:
        ``(len(ids),)`` or ``(len(ids), k)`` float array aligned with ``ids``.

    Raises:
        ValueError: requested ids missing, extra rows in the file (unless
            ``allow_subset``), duplicate ids, or a legacy ``.npy`` whose
            length does not match ``ids``.
        FileNotFoundError: neither a parquet nor a legacy ``.npy`` exists.
    """
    ids = np.asarray(ids)
    frame = _read(results_dir, name, split)
    if frame is not None:
        preds = _aligned(
            frame,
            ids,
            id_col,
            "pred",
            f"{split}_{name}.parquet",
            allow_subset=allow_subset,
        ).to_numpy(dtype=np.float64)
        return preds[:, 0] if preds.shape[1] == 1 else preds

    legacy = results_dir / f"{split}_{name}.npy"
    if not legacy.exists():
        msg = f"no {split}_{name}.parquet or .npy in {results_dir}"
        raise FileNotFoundError(msg)
    arr: NDArray[np.float64] = np.load(legacy).astype(np.float64)
    if len(arr) != len(ids):
        msg = f"{legacy.name}: {len(arr)} rows but {len(ids)} ids requested"
        raise ValueError(msg)
    warnings.warn(
        f"{legacy.name} is a positional legacy array: row alignment cannot be verified "
        "(ids must be in build_features() order)",
        stacklevel=2,
    )
    return arr


def load_folds(
    results_dir: Path, name: str, ids: NDArray[Any], *, id_col: str = "id"
) -> NDArray[np.int64]:
    """Validation fold of each id, as stored in ``oof_<name>.parquet``.

    Use it to check two runs share folds before stacking them:
    ``(load_folds(r, "a", ids) == load_folds(r, "b", ids)).all()``.
    """
    frame = _read(results_dir, name, "oof")
    if frame is None or "fold" not in frame:
        msg = f"oof_{name}.parquet with a 'fold' column not found in {results_dir}"
        raise FileNotFoundError(msg)
    folds = _aligned(
        frame, np.asarray(ids), id_col, "fold", f"oof_{name}.parquet", allow_subset=True
    )
    return folds["fold"].to_numpy(dtype=np.int64)


def list_predictions(results_dir: Path) -> list[str]:
    """Names with both an OOF and a test file (parquet or legacy npy), sorted."""
    names: set[str] = set()
    for path in results_dir.glob("oof_*"):
        if path.suffix not in {".parquet", ".npy"}:
            continue
        stem = path.stem[len("oof_") :]
        if any(
            (results_dir / f"test_{stem}{ext}").exists() for ext in (".parquet", ".npy")
        ):
            names.add(stem)
    return sorted(names)


def migrate_npy(
    results_dir: Path,
    name: str,
    *,
    train_ids: NDArray[Any],
    test_ids: NDArray[Any],
    folds: NDArray[Any] | None = None,
    id_col: str = "id",
) -> PredictionPaths:
    """Convert a legacy ``oof_/test_<name>.npy`` pair to keyed parquet.

    ``train_ids`` / ``test_ids`` must be in the order the arrays were written
    (``build_features()`` order) — this is the one place that trust is needed;
    afterwards the ids travel with the predictions. The ``.npy`` files are kept.
    """
    oof = np.load(results_dir / f"oof_{name}.npy")
    test = np.load(results_dir / f"test_{name}.npy")
    return save_predictions(
        results_dir,
        name,
        oof=oof,
        test=test,
        train_ids=train_ids,
        test_ids=test_ids,
        folds=folds,
        id_col=id_col,
    )
