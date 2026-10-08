"""PR-18: learning curve of the tuned LGBM route+orig leg (does more data still pay?).

    python src/lc_screen.py --folds 3 --fracs 0.5 0.75 1.0

For each of the first `--folds` CV folds (same splitter as run_cv), train on a
random fraction of the fold's synthetic training rows (original rows always all
appended; route TE fit on the kept rows only) and score the untouched val fold.
The slope of AUC vs log2(n) estimates what a 100%-data refit adds over the
80%-data fold models that produce today's test predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from matrices import build_matrices
from route import add_route_global, route_te
from train_lgbm_orig import PARAMS as CONTROL_PARAMS

from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    p.add_argument("--n-jobs", type=int, default=10)
    a = p.parse_args()

    tuned = json.loads((RESULTS_DIR / "best_params_lgbm_pr5.json").read_text())
    params = {**CONTROL_PARAMS, **tuned, "n_jobs": a.n_jobs}
    m = build_matrices(with_original=True)
    n_test = len(m.X_test)
    m.X_test = pd.concat([m.X_test, m.X_orig], ignore_index=True)
    add_route_global(m)
    x_orig = m.X_test.iloc[n_test:].reset_index(drop=True)
    assert m.y_orig is not None

    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
    rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        if fold > a.folds:
            break
        rng = np.random.default_rng(fold)
        perm = rng.permutation(tr)
        for frac in a.fracs:
            keep = np.sort(perm[: int(round(frac * len(tr)))])
            x_tr, y_tr = m.X.iloc[keep], m.y[keep]
            x_va, y_va = m.X.iloc[va], m.y[va]
            x_tr, x_va, x_o = route_te(x_tr, y_tr, [x_va, x_orig])
            x_tr = pd.concat([x_tr, x_o[x_tr.columns]], ignore_index=True)
            y_fit = np.concatenate([y_tr, m.y_orig])
            model = LGBMClassifier(**params)
            model.fit(x_tr, y_fit, eval_set=[(x_va, y_va)],
                      callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
            auc = float(roc_auc_score(y_va, model.predict_proba(x_va)[:, 1]))
            rows.append((fold, frac, len(keep), auc, model.best_iteration_))
            print(f"fold {fold} frac {frac:.2f} n={len(keep)} AUC {auc:.6f} "
                  f"best_iter {model.best_iteration_}", flush=True)
    df = pd.DataFrame(rows, columns=["fold", "frac", "n", "auc", "iter"])
    # Per-fold slope of AUC on log2(n); paired within fold.
    slopes = [np.polyfit(np.log2(g["n"]), g["auc"], 1)[0] for _, g in df.groupby("fold")]
    s = float(np.mean(slopes))
    print(df.pivot(index="fold", columns="frac", values="auc"))
    print(f"slope per doubling {s:+.6f} (folds {np.round(slopes, 6).tolist()})")
    print(f"extrapolated 80%→100% refit gain ≈ {s * np.log2(1.25):+.6f}")


if __name__ == "__main__":
    main()
