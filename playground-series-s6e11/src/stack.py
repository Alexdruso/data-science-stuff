"""LR-on-logits stack over saved legs.

    python src/stack.py NAME1 NAME2 ... [--C 1.0] [--out stack_v1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import load_predictions, save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    ids, y = train["id"].to_numpy(), train[TARGET].to_numpy().astype(int)
    oofs = np.column_stack([np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel() for n in a.names])
    tests = np.column_stack(
        [np.asarray(load_predictions(RESULTS_DIR, n, test_ids, split="test")).ravel() for n in a.names])
    for n, col in zip(a.names, oofs.T):
        print(f"  {n:30s} {roc_auc_score(y, col):.5f}")
    Xo, Xt = logit(oofs), logit(tests)
    print(f"  mean-logit blend               {roc_auc_score(y, Xo.mean(1)):.5f}")
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(test_ids))
    seeds = [7, 13, 99]
    for s in seeds:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Xo, y):
            m = LogisticRegression(C=a.C, max_iter=2000).fit(Xo[tr], y[tr])
            meta_oof[va] += m.decision_function(Xo[va]) / len(seeds)
            meta_test += m.decision_function(Xt) / (5 * len(seeds))
    auc = float(roc_auc_score(y, meta_oof))
    full = LogisticRegression(C=a.C, max_iter=2000).fit(Xo, y)
    print("  weights", np.round(full.coef_[0], 3))
    print(f"STACK OOF AUC {auc:.5f}")
    if a.out:
        prob = 1 / (1 + np.exp(-meta_test))
        save_predictions(RESULTS_DIR, a.out, oof=1 / (1 + np.exp(-meta_oof)), test=prob,
                         train_ids=ids, test_ids=test_ids)
        save_cv_result(RESULTS_DIR, a.out, [], auc, metric_name="auc",
                       params={"members": a.names, "C": a.C})
        write_submission(SUBMISSIONS_DIR, f"{a.out}.csv", test_ids, TARGET, prob)


if __name__ == "__main__":
    main()
