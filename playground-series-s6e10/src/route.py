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


def add_extras(m: Matrices, x_orig: pd.DataFrame | None, extras: list[str]) -> None:
    """PR-19/20 label-free extras, applied after the original rows are split off X_test.

    og   — original-data teacher logits (`og_model.py`; original rows get OOF logits)
    agg  — route profile: per-fd mean of features over synthetic train + test
    aux  — auxiliary-task expected ratings (`aux_feats.py`); `auxall` = all aux columns
    """
    if "og" in extras:
        from og_model import add_og
        add_og(m, x_orig)
    if "agg" in extras:
        from run_lgbm_arm import add_route_agg
        n = len(m.X_test)
        if x_orig is not None:
            m.X_test = pd.concat([m.X_test, x_orig], ignore_index=True)
        # Means over synthetic rows only; original rows just receive their route's values.
        syn = Matrices(X=m.X, y=m.y, X_test=m.X_test.iloc[:n].copy(), test_ids=m.test_ids,
                       feature_cols=m.feature_cols, cat_cols=m.cat_cols)
        both = pd.concat([syn.X, syn.X_test])
        add_route_agg(syn)
        new = [c for c in syn.X.columns if c.startswith("fdmean_")]
        means = pd.concat([syn.X[new], syn.X_test[new]]).groupby(both["flight_distance"].to_numpy()).first()
        m.X = syn.X
        tail = m.X_test.iloc[n:]
        m.X_test = syn.X_test
        if x_orig is not None:
            vals = means.reindex(tail["flight_distance"].to_numpy()).to_numpy()
            for i, c in enumerate(new):
                x_orig[c] = vals[:, i]
    if "aux" in extras or "auxall" in extras:
        from aux_feats import add_aux
        add_aux(m, x_orig, which="all" if "auxall" in extras else "ev")
