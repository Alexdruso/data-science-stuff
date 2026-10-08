"""PR-22: export fold-ready float matrices for a tabular foundation model (TabICL).

    python src/export_tfm.py            # → results/tfm_fold{k}.npz, k = 1..5

Per fold (run_cv's splitter): route TE fit on the fold's training split only,
plus the label-free extras (teacher logits, route profile, aux expected ratings).
The TFM runs in an isolated venv (`/tmp/tabicl_venv`), so it gets plain arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from matrices import build_matrices
from route import add_extras, add_route_global, route_te

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)


def _num(f: pd.DataFrame) -> np.ndarray:
    out = f.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.codes
    return out.to_numpy(np.float32)


def main() -> None:
    m = build_matrices()
    add_route_global(m)
    add_extras(m, None, ["og", "agg", "aux"])
    X = m.X.drop(columns=["is_original"])
    X_test = m.X_test.drop(columns=["is_original"])
    cat_idx = [i for i, c in enumerate(X.columns) if isinstance(X[c].dtype, pd.CategoricalDtype)]
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, m.y)
    for k, (tr, va) in enumerate(splits, 1):
        x_tr, x_va, x_te = route_te(X.iloc[tr], m.y[tr], [X.iloc[va], X_test])
        np.savez(RESULTS_DIR / f"tfm_fold{k}.npz", x_tr=_num(x_tr), y_tr=m.y[tr], x_va=_num(x_va),
                 y_va=m.y[va], va_idx=va, x_te=_num(x_te), cols=np.array(list(x_tr.columns)),
                 cat_idx=np.array(cat_idx))
        print(f"fold {k}: {x_tr.shape} {x_va.shape} {x_te.shape}", flush=True)


if __name__ == "__main__":
    main()
