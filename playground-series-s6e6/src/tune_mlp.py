"""Optuna hyperparameter search for the MLP — PS S6E6.

3-fold CV per trial (objective = mean fold argmax balanced accuracy); saves best
params to results/best_params_mlp.json. Reuses train_mlp.train_fold / prepare_arrays
so the search and the final retrain share identical per-fold logic.

GPU-memory rule (CLAUDE.md): train_fold dels its tensors; this loop calls
torch.cuda.empty_cache() after every fold and trial — required for a 150-iteration
(50 trials × 3 folds) run not to fragment the 6 GB card.

Run with:  python src/tune_mlp.py [--n-trials N] [--timeout SECS]
After tuning, re-run src/train_mlp.py — it auto-loads the JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from train_mlp import DEVICE, prepare_arrays, train_fold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna not installed — run: uv pip install optuna") from e

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_TUNE_FOLDS = 3


def objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    y: np.ndarray,
    num_idx: list[int],
    cat_idx: list[int],
) -> float:
    params: dict[str, object] = {
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "first_size": trial.suggest_categorical("first_size", [256, 512, 768, 1024]),
        "shrink": trial.suggest_float("shrink", 0.3, 0.8),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
    }

    skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=42)
    fold_scores: list[float] = []
    for train_idx, val_idx in skf.split(X, y):
        # Dummy 2-row X_test — train_fold predicts on it but we ignore the result.
        val_pred, _ = train_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X[val_idx][:2], num_idx, cat_idx, params,
        )
        fold_scores.append(
            float(balanced_accuracy_score(y[val_idx], np.argmax(val_pred, axis=1)))
        )
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(fold_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    X, _X_test, num_idx, cat_idx = prepare_arrays(train_pl, test_pl, cat_cols, feature_cols)
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    print(f"Feature matrix: {X.shape}  classes: {list(le.classes_)}  device: {DEVICE}")

    study = optuna.create_study(direction="maximize", study_name="mlp_s6e6")
    study.optimize(
        lambda t: objective(t, X, y, num_idx, cat_idx),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: {best.value:.4f}")
    print("Params:", json.dumps(best.params, indent=2))

    out_path = RESULTS_DIR / "best_params_mlp.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved → {out_path}")
    print("Re-run src/train_mlp.py to train with tuned params.")


if __name__ == "__main__":
    main()
