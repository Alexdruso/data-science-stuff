"""Optuna on XGB (GPU), scored on folds 1-2 of the standard split. Appends trials to
results/tune_xgb.jsonl; best params -> results/best_params_xgb.json.

    python src/tune_xgb.py --trials 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features
from train_gbdt import to_pandas

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    feats = [c for c in train.columns if c not in EXCLUDE_COLS]
    (tr_pd,) = to_pandas([train])
    X, y = tr_pd[feats], tr_pd[TARGET].to_numpy()
    folds = fold_assignment(y)

    def objective(trial: optuna.Trial) -> float:
        p = {"max_depth": trial.suggest_int("max_depth", 2, 5),
             "learning_rate": 0.03,
             "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50, log=True),
             "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10, log=True),
             "gamma": trial.suggest_float("gamma", 0.0, 2.0),
             "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256, 512]),
             "max_cat_to_onehot": trial.suggest_categorical("max_cat_to_onehot", [1, 16])}
        scores = []
        for k in (1, 2):
            tr, va = folds != k, folds == k
            m = xgb.XGBClassifier(n_estimators=8000, tree_method="hist", device="cuda",
                                  enable_categorical=True, eval_metric="auc",
                                  early_stopping_rounds=200, n_jobs=4, random_state=42, **p)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            trial.set_user_attr(f"iter{k}", int(m.best_iteration))
            del m
        s = float(np.mean(scores))
        with open(RESULTS_DIR / "tune_xgb.jsonl", "a") as f:
            f.write(json.dumps({"score": s, **p, **trial.user_attrs}) + "\n")
        print(f"trial {trial.number}: {s:.5f} {p}", flush=True)
        return s

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({"max_depth": 2, "min_child_weight": 5, "subsample": 0.8,
                      "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.001,
                      "gamma": 0.0, "max_bin": 256, "max_cat_to_onehot": 1})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, n_trials=a.trials)
    best = {**st.best_params, "learning_rate": 0.03}
    (RESULTS_DIR / "best_params_xgb.json").write_text(json.dumps(best, indent=1))
    print("best", st.best_value, best)


if __name__ == "__main__":
    main()
