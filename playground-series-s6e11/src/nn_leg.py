"""pytabkit NN legs (RealMLP-TD default) on raw features, 5-fold CV.

    python src/nn_leg.py --model realmlp [--seed 42]

pytabkit early-stops on its own split of the TRAINING rows, so the scored fold never
selects the epoch (no OOF inflation).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import CAT_COLS, EXCLUDE_COLS, TARGET, build_features
from train_gbdt import to_pandas

from data_science_stuff.kaggle.cv import fold_assignment, run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import save_predictions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def make(model: str, seed: int) -> Any:
    if model == "realmlp":
        from pytabkit import RealMLP_TD_Classifier

        return RealMLP_TD_Classifier(device="cuda", random_state=seed, n_cv=1, n_epochs=48, batch_size=1024,
                                     val_metric_name="cross_entropy", verbosity=0)
    if model == "tabm":
        from pytabkit import TabM_D_Classifier

        return TabM_D_Classifier(device="cuda", random_state=seed, n_cv=1,
                                 val_metric_name="cross_entropy", batch_size=1024, verbosity=0)
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="realmlp")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    name = f"{a.model}_s{a.seed}"
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    feats = [c for c in train.columns if c not in EXCLUDE_COLS]
    tr_pd, te_pd = to_pandas([train, test])
    X, X_test = tr_pd[feats], te_pd[feats]
    y = tr_pd[TARGET].to_numpy().astype(int)

    def fit_fold(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray,
                 x_te: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
        m = make(a.model, a.seed + fold)
        m.fit(x_tr, y_tr, cat_col_names=CAT_COLS)
        va, te = m.predict_proba(x_va)[:, 1], m.predict_proba(x_te)[:, 1]
        print(f"  fold {fold}: auc {roc_auc_score(y_va, va):.5f}", flush=True)
        del m
        return va, te

    res = run_cv(X, y, X_test, fit_fold, score_fn=lambda u, v: float(roc_auc_score(u, v)),
                 after_fold=lambda _f: torch.cuda.empty_cache())
    auc = float(roc_auc_score(y, res.oof))
    print(f"{name}: OOF AUC {auc:.5f} folds {np.round(res.fold_scores, 5)}")
    save_predictions(RESULTS_DIR, name, oof=res.oof, test=res.test,
                     train_ids=tr_pd["id"].to_numpy(), test_ids=te_pd["id"].to_numpy(),
                     folds=fold_assignment(y))
    save_cv_result(RESULTS_DIR, name, res.fold_scores, auc, metric_name="auc",
                   params={"model": a.model, "seed": a.seed})
    write_submission(SUBMISSIONS_DIR, f"{name}.csv", te_pd["id"].to_numpy(), TARGET, res.test)


if __name__ == "__main__":
    main()
