"""Compare an LM pilot scored on a val subset against the saved legs on the same rows.

    python src/lm_sub_eval.py _lm_distilgpt2_p2 [--fold 1]
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

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()
    d = RESULTS_DIR / a.dir
    idx = np.load(d / f"f{a.fold}_valsub_idx.npy")
    lm = np.load(d / f"f{a.fold}_valsub.npy")
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    ys = y[idx]
    print(f"rows {len(idx)}  pos {ys.sum()}")
    print(f"  lm                 {roc_auc_score(ys, lm):.5f}")
    legs = {n: logit(np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel()[idx])
            for n in ["ebm_i30", "ebm_i30_s7", "xgbt2_b2048", "stack_v2"]}
    for n, v in legs.items():
        print(f"  {n:18s} {roc_auc_score(ys, v):.5f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    base = np.column_stack([legs["ebm_i30"], legs["ebm_i30_s7"], legs["xgbt2_b2048"]])
    for nm, X in [("base3", base), ("base3+lm", np.column_stack([base, lm]))]:
        p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), X, ys, cv=cv,
                              method="decision_function")
        print(f"  stack {nm:12s} {roc_auc_score(ys, p):.5f}")
    print("  corr(lm, ebm_i30)", np.corrcoef(lm, legs["ebm_i30"])[0, 1].round(4))


if __name__ == "__main__":
    main()
