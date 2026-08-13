"""Arms `s7`/`s8` — quantile-bin target encoding, fold-safe.

The Day-10 residual screen (`screen_te_crosses.py`) said "no uncaptured additive
structure in these cells". That is a proxy, not the deliverable: a retrained model
gets the TE column as a new *axis* it can split on and interact with every other
feature, which an additive residual correction cannot represent. So the arms get
built and read directly.

Two arms, because the proposal has two distinct forms:

* `s7` — TE of the six pre-registered 2-way **crosses**. The class with a real
  prior: a k-way cell mean needs tree depth >= k AND the interaction to survive
  regularisation.
* `s8` — TE of **single** quantile-binned continuous columns. Expected flat on
  theory (``E[y | qbin(x)]`` is what a depth-1 tree on ``x`` already outputs, so
  it is a coarsening), included so that expectation is tested rather than
  asserted.

Fold safety (CLAUDE.md leakage rule 1 and 2): both transforms run INSIDE the fold
loop via the `ARMS` table, so quantile edges come from the fold's training split
only (`qcut_codes(..., ref_values=x_tr)`) and every target mean is fit by
`add_fold_safe_target_encoding`, which additionally uses nested CV on the training
frame so the encoding a row sees was never fit on that row's own label.

Row order is untouched — these transforms only add columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data_science_stuff.kaggle.encoding import add_fold_safe_target_encoding, qcut_codes

Frames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

#: `{1: name}` — binary target, so one TE column per source column.
_CLASSES = {1: "pos"}

#: (cross name, col_a, col_b, bins_a, bins_b) — the six from CLAUDE.md item 10.
CROSSES: list[tuple[str, str, str, int, int]] = [
    ("soc_share_x_daily", "social_share", "daily_screen_time_hours", 16, 16),
    ("nobs_x_sgwlb", "n_obs_sgw", "sgw_lb", 4, 16),
    ("blknulls_x_sgwlb", "screen_block_nulls", "sgw_lb", 5, 16),
    ("awake_x_sleep", "screen_share_of_awake", "sleep_hours", 16, 16),
    ("gam_share_x_soc_share", "gaming_share", "social_share", 16, 16),
    ("stress_x_daily", "stress_ord", "daily_screen_time_hours", 3, 16),
]

#: Single columns for `s8`, at fine granularity (q=64) to give TE its best shot.
SINGLES: list[str] = [
    "social_share",
    "gaming_share",
    "daily_screen_time_hours",
    "social_media_hours",
    "weekend_screen_time",
    "screen_share_of_awake",
    "notifications_per_open",
    "sgw_lb",
]
SINGLE_Q = 64


def _codes(frame: pd.DataFrame, ref: pd.DataFrame, col: str, q: int) -> NDArray[np.int16]:
    """Quantile codes for `col`, edges from the fold's training split only."""
    return qcut_codes(
        frame[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float), q
    )


def _add_cells(
    frames: Frames, ref: pd.DataFrame, spec: list[tuple[str, str, str, int, int]]
) -> tuple[Frames, list[str]]:
    """Add cross-cell string keys to every frame; return the new column names."""
    out = tuple(f.copy() for f in frames)
    names = []
    for name, a, b, qa, qb in spec:
        for frame in out:
            ca = _codes(frame, ref, a, qa)
            cb = _codes(frame, ref, b, qb)
            frame[name] = pd.Series(
                np.char.add(np.char.add(ca.astype(str), "_"), cb.astype(str)),
                index=frame.index,
                dtype="object",
            )
        names.append(name)
    return out, names  # type: ignore[return-value]


def _add_single_cells(
    frames: Frames, ref: pd.DataFrame, cols: list[str], q: int
) -> tuple[Frames, list[str]]:
    """Add single-column quantile-bin string keys to every frame."""
    out = tuple(f.copy() for f in frames)
    names = []
    for col in cols:
        name = f"{col}_qb{q}"
        for frame in out:
            frame[name] = pd.Series(
                _codes(frame, ref, col, q).astype(str), index=frame.index, dtype="object"
            )
        names.append(name)
    return out, names  # type: ignore[return-value]


def _encode_and_drop(
    frames: Frames, y_tr: NDArray[np.int64], cells: list[str]
) -> Frames:
    """Target-encode the cell columns fold-safely, then drop the raw cells."""
    x_tr, x_va, x_te = frames
    enc_tr, enc_va, enc_te = add_fold_safe_target_encoding(
        x_tr, y_tr, [x_va, x_te], cells, _CLASSES, smooth=20.0, inner_cv=5
    )
    # The cells themselves are high-cardinality and are NOT the lever under test.
    return (
        enc_tr.drop(columns=cells),
        enc_va.drop(columns=cells),
        enc_te.drop(columns=cells),
    )


def add_te_crosses(
    x_tr: pd.DataFrame, y_tr: NDArray[np.int64], x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s7`: `s1` + fold-safe TE of the six pre-registered 2-way crosses."""
    frames, cells = _add_cells((x_tr, x_va, x_te), x_tr, CROSSES)
    return _encode_and_drop(frames, y_tr, cells)


def add_te_singles(
    x_tr: pd.DataFrame, y_tr: NDArray[np.int64], x_va: pd.DataFrame, x_te: pd.DataFrame
) -> Frames:
    """Arm `s8`: `s1` + fold-safe TE of eight single quantile-binned columns."""
    frames, cells = _add_single_cells((x_tr, x_va, x_te), x_tr, SINGLES, SINGLE_Q)
    return _encode_and_drop(frames, y_tr, cells)
