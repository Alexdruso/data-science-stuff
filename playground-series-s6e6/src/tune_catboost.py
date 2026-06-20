"""Optuna hyperparameter search for CatBoost — PS S6E6.

Runs 3-fold CV per trial for speed; saves best params to
results/best_params_catboost.json. GPU-accelerated (task_type="GPU").

Imbalance is handled exactly as in train_catboost.py — auto_class_weights="Balanced"
in the fixed params, and categoricals are passed via Pool(cat_features=...).

Run with:  python src/tune_catboost.py [--n-trials N] [--timeout SECS]
After tuning, re-run src/train_catboost.py — it auto-loads the JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna not installed — run: uv pip install optuna") from e

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_TUNE_FOLDS = 3
# Fixed params not part of the search space (GPU/objective stay authoritative;
# train_catboost.py's CB_PARAMS.update(tuned) must not collide).
FIXED_PARAMS: dict[str, object] = {
    "iterations": 1000,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "auto_class_weights": "Balanced",
    "task_type": "GPU",
    "random_seed": 42,
    "verbose": 0,
}


def objective(
    trial: "optuna.Trial",
    X: "pl.DataFrame",  # pandas DataFrame at call time
    y: np.ndarray,
    cat_indices: list[int],
) -> float:
    params: dict[str, object] = {
        **FIXED_PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 254),
    }

    skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=42)
    fold_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        val_pred = np.argmax(model.predict_proba(val_pool), axis=1)
        fold_scores.append(float(balanced_accuracy_score(y_val, val_pred)))

    return float(np.mean(fold_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train_pl = build_features(train_raw)
    train_pl = compute_group_features(train_raw, train_pl)

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    cat_indices = [feature_cols.index(c) for c in cat_cols]

    train_pd = train_pl.to_pandas()
    X = train_pd[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"Feature matrix: {X.shape}  classes: {list(le.classes_)}")
    print("CatBoost task_type: GPU")

    study = optuna.create_study(direction="maximize", study_name="catboost_s6e6")
    study.optimize(
        lambda trial: objective(trial, X, y, cat_indices),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: {best.value:.4f}")
    print("Params:", json.dumps(best.params, indent=2))

    out_path = RESULTS_DIR / "best_params_catboost.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved → {out_path}")
    print("Re-run src/train_catboost.py to train with tuned params.")


if __name__ == "__main__":
    main()
