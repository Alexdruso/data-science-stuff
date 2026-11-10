"""Fold-level read of LM variants: solo AUC and the gain each adds to the EBM/tree stack.

    python src/lm_fold_eval.py _lm_distilgpt2 _lm_distilgpt2_orig10 [--fold 1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
BASE = ["ebm_i30_s7", "ebm_i30_mib256", "ebm_i30_s7_mib256", "ebm_i30", "ebm_i60", "xgbt2_b2048",
        "xgbt2_b2048_s1", "lgbm_d2_b2048"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    m = fold_assignment(y) == a.fold
    ys = y[m]
    base = np.column_stack([logit(np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel()[m])
                            for n in BASE])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def stack(x: np.ndarray) -> float:
        p = cross_val_predict(LogisticRegression(max_iter=3000), x, ys, cv=cv,
                              method="decision_function")
        return float(roc_auc_score(ys, p))

    b = stack(base)
    print(f"fold {a.fold}: base stack {b:.5f}")
    lms = {}
    for d in a.dirs:
        lm = np.load(RESULTS_DIR / d / f"f{a.fold}_val.npy")
        lms[d] = lm
        s = stack(np.column_stack([base, lm]))
        print(f"  {d:28s} solo {roc_auc_score(ys, lm):.5f}  stack {s:.5f} ({s - b:+.5f})")
    if len(lms) > 1:
        s = stack(np.column_stack([base, *lms.values()]))
        print(f"  all LMs together               stack {s:.5f} ({s - b:+.5f})")


if __name__ == "__main__":
    main()
