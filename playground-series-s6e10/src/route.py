"""Route features for PS S6E10 (PR-12) — shared by every leg.

`flight_distance` takes only 3,521 values, all copied from the original dataset,
and behaves like a route ID. Two parts:

* `add_route_global(m)` — label-free columns on X / X_test: the original's
  smoothed satisfaction rate + row count per distance, the synthetic value
  count, and the string route keys that `route_te` encodes;
* `route_te(x_tr, y_tr, others)` — call INSIDE each fold: fold-safe TE of the
  route keys (fit on the fold's training split only), keys dropped afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from matrices import Matrices
from run_lgbm_arm import TE_FD_COLS, add_route

from data_science_stuff.kaggle.encoding import add_fold_safe_target_encoding


def add_route_global(m: Matrices) -> None:
    m.X, m.X_test = m.X.copy(), m.X_test.copy()
    add_route(m)


def route_te(
    x_tr: pd.DataFrame, y_tr: np.ndarray, others: list[pd.DataFrame]
) -> tuple[pd.DataFrame, ...]:
    frames = add_fold_safe_target_encoding(x_tr, y_tr, others, TE_FD_COLS, {1: "sat"})
    return tuple(f.drop(columns=TE_FD_COLS) for f in frames)
