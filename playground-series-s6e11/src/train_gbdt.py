"""GBDT legs for PS S6E11 (LightGBM / XGBoost / CatBoost), 5-fold stratified CV.

    python src/train_gbdt.py --model lgbm [--orig] [--seed 42] [--name NAME]

`--orig` appends the 20k original rows to every training fold (never scored).
Predictions are saved as keyed parquet (`results/oof_<name>.parquet`).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import CAT_COLS, EXCLUDE_COLS, TARGET, add_extra, build_features, load_original

from data_science_stuff.kaggle.cv import fold_assignment, run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import save_predictions

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_JOBS = 8  # keep the box below full load (it hard-reboots under heavy load)


def to_pandas(frames: list[pl.DataFrame]) -> list[pd.DataFrame]:
    pds = [f.to_pandas() for f in frames]
    for c in CAT_COLS:
        levels = sorted(set().union(*[set(p[c].unique()) for p in pds]))
        for p in pds:
            p[c] = pd.Categorical(p[c], categories=levels)
    return pds


def make_fit(model: str, seed: int, params: dict[str, Any]) -> Any:
    def fit(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray,
            x_te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
        if model == "lgbm":
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation

            p = {"objective": "binary", "n_estimators": 10000, "learning_rate": 0.02,
                 "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.5, "reg_lambda": 1.0,
                 "max_bin": 255, "verbose": -1, "n_jobs": N_JOBS, "random_state": seed}
            p.update(params)
            m = LGBMClassifier(**p)
            m.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], eval_metric="auc",
                  callbacks=[early_stopping(300, verbose=False), log_evaluation(0)])
            return m.predict_proba(x_va)[:, 1], m.predict_proba(x_te)[:, 1], int(m.best_iteration_)
        if model == "xgb":
            import xgboost as xgb

            p = {"n_estimators": 10000, "learning_rate": 0.02, "max_depth": 6,
                 "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.5,
                 "reg_lambda": 1.0, "tree_method": "hist", "device": "cuda",
                 "enable_categorical": True, "max_cat_to_onehot": 1,
                 "eval_metric": "auc", "early_stopping_rounds": 300,
                 "random_state": seed, "n_jobs": N_JOBS}
            p.update(params)
            m = xgb.XGBClassifier(**p)
            m.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
            out = m.predict_proba(x_va)[:, 1], m.predict_proba(x_te)[:, 1], int(m.best_iteration)
            del m
            return out
        if model == "cat":
            from catboost import CatBoostClassifier

            p = {"iterations": 10000, "learning_rate": 0.04, "depth": 6,
                 "eval_metric": "AUC", "od_type": "Iter", "od_wait": 300,
                 "random_seed": seed, "verbose": 0, "thread_count": N_JOBS}
            p.update(params)
            def cat(x: pd.DataFrame) -> pd.DataFrame:
                x = x.copy()
                for c in CAT_COLS:
                    x[c] = x[c].astype(str)
                return x
            m = CatBoostClassifier(**p, cat_features=CAT_COLS)
            m.fit(cat(x_tr), y_tr, eval_set=(cat(x_va), y_va), use_best_model=True)
            return (m.predict_proba(cat(x_va))[:, 1], m.predict_proba(cat(x_te))[:, 1],
                    int(m.get_best_iteration()))
        raise ValueError(model)
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--orig", action="store_true")
    ap.add_argument("--fe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--params", default="{}", help="JSON overrides")
    ap.add_argument("--prereg", default=None)
    args = ap.parse_args()
    params = json.loads(args.params)
    name = args.name or f"{args.model}{'_orig' if args.orig else ''}_s{args.seed}"

    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    orig = build_features(load_original(DATA_DIR))
    if args.fe:
        train, test, orig = add_extra([train, test, orig], orig)
    feats = [c for c in train.columns if c not in EXCLUDE_COLS]
    tr_pd, te_pd, or_pd = to_pandas([train, test, orig])
    X, X_test, X_orig = tr_pd[feats], te_pd[feats], or_pd[feats]
    y = tr_pd[TARGET].to_numpy()
    y_orig = or_pd[TARGET].to_numpy()
    fit = make_fit(args.model, args.seed, params)
    iters: list[int] = []

    def fit_fold(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame,
                 y_va: np.ndarray, x_te: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
        if args.orig:
            x_tr = pd.concat([x_tr, X_orig], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])
        va, te, it = fit(x_tr, y_tr, x_va, y_va, x_te)
        iters.append(it)
        print(f"  fold {fold}: auc {roc_auc_score(y_va, va):.5f} iters {it}", flush=True)
        return va, te

    def after(_f: int) -> None:
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    res = run_cv(X, y, X_test, fit_fold, score_fn=lambda a, b: float(roc_auc_score(a, b)),
                 after_fold=after)
    auc = float(roc_auc_score(y, res.oof))
    print(f"{name}: OOF AUC {auc:.5f}  folds {np.round(res.fold_scores, 5)}  iters {iters}")
    save_predictions(RESULTS_DIR, name, oof=res.oof, test=res.test,
                     train_ids=tr_pd["id"].to_numpy(), test_ids=te_pd["id"].to_numpy(),
                     folds=fold_assignment(y))
    save_cv_result(RESULTS_DIR, name, res.fold_scores, auc, metric_name="auc",
                   params={"model": args.model, "orig": args.orig, "fe": args.fe, "seed": args.seed, **params},
                   **({"prereg": args.prereg} if args.prereg else {}))
    write_submission(SUBMISSIONS_DIR, f"{name}.csv", te_pd["id"].to_numpy(), TARGET, res.test)


if __name__ == "__main__":
    main()
