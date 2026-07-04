"""PyTorch MLP 5-fold CV training for PS S6E6 — Predicting Stellar Class.

3-class classifier (GALAXY/QSO/STAR) — the primary non-GBDT diversity bet for
the ensemble. The model, per-fold training loop, and OHE feature matrix now
come from data_science_stuff.kaggle.models.mlp; this driver keeps the
competition specifics (feature building, LabelEncoder, thresholds, saving)
plus thin wrappers preserving the names tune_mlp.py / train_mlp_la.py /
train_realmlp_emb.py import.

GPU-memory rule (CLAUDE.md): fit_mlp_fold dels its model/optimizer/tensors and
the run_cv after_fold hook calls torch.cuda.empty_cache() after each fold.

Run with:  python src/train_mlp.py
Auto-loads results/best_params_mlp.json if present (from tune_mlp.py).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission
from data_science_stuff.kaggle.models.mlp import (
    MLP,
    fit_mlp_fold,
    ohe_feature_matrix,
)
from data_science_stuff.kaggle.models.mlp import (
    build_layer_sizes as _build_layer_sizes,
)
from data_science_stuff.kaggle.models.mlp import (
    class_weights as _class_weights,
)

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

N_FOLDS = 5
N_CLASSES = 3
BATCH_SIZE = 8192
EPOCHS = 100
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PARAMS: dict[str, object] = {
    "n_layers": 3,
    "first_size": 512,
    "shrink": 0.5,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "activation": "relu",
}

__all__ = [
    "BATCH_SIZE", "DEFAULT_PARAMS", "DEVICE", "EPOCHS", "MLP", "N_CLASSES",
    "N_FOLDS", "PATIENCE", "build_layer_sizes", "class_weights",
    "prepare_arrays", "train_fold",
]


def build_layer_sizes(params: dict[str, object]) -> list[int]:
    """Dict-shaped wrapper over the package helper (tune_mlp compatibility)."""
    return _build_layer_sizes(
        int(params["n_layers"]),  # type: ignore[arg-type]
        int(params["first_size"]),  # type: ignore[arg-type]
        float(params.get("shrink", 0.5)),  # type: ignore[arg-type]
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, sklearn 'balanced' formula: n / (k * count)."""
    return _class_weights(y, N_CLASSES)


def prepare_arrays(
    train_pl: pl.DataFrame,
    test_pl: pl.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """OHE categoricals; return (X, X_test, num_idx, cat_idx) as float32 matrices."""
    return ohe_feature_matrix(
        train_pl.to_pandas(), test_pl.to_pandas(), cat_cols, feature_cols
    )


def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    num_idx: list[int],
    cat_idx: list[int],
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """One fold via the shared fit_mlp_fold (per-fold scaling, early stopping)."""
    return fit_mlp_fold(
        X_tr, y_tr, X_val, y_val, X_test, num_idx, cat_idx, params,
        n_classes=N_CLASSES, device=DEVICE,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
    )


def main() -> None:
    print(f"Device: {DEVICE}")
    params = load_params(RESULTS_DIR, DEFAULT_PARAMS, "best_params_mlp.json")
    print(f"MLP params: {params}")

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    print(f"Train: {train_pl.shape}   Test: {test_pl.shape}")

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    X, X_test, num_idx, cat_idx = prepare_arrays(
        train_pl, test_pl, cat_cols, feature_cols
    )
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    test_ids = test_pl.to_pandas()["id"].to_numpy()
    print(f"Feature matrix: {X.shape}  classes: {list(le.classes_)}")

    def fit_fold(x_tr, y_tr, x_va, y_va, x_te, _fold):
        return train_fold(x_tr, y_tr, x_va, y_va, x_te, num_idx, cat_idx, params)

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(balanced_accuracy_score(y_va, np.argmax(val_pred, axis=1)))

    def free_gpu(_fold: int) -> None:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    oof_proba, test_proba, fold_scores = run_cv(
        X, y, X_test, fit_fold,
        n_splits=N_FOLDS, n_outputs=N_CLASSES,
        score_fn=fold_score, after_fold=free_gpu,
    )
    for fold, score in enumerate(fold_scores, 1):
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    oof_score = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))
    print(f"\nOOF balanced_acc (argmax): {oof_score:.4f}")

    threshold_weights, best_score = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    save_threshold_weights(
        threshold_weights, le.classes_.tolist(),
        RESULTS_DIR / "threshold_weights_mlp.json",
    )
    save_cv_result(RESULTS_DIR, "mlp_v1", fold_scores, best_score, metric_name="balanced_acc")

    np.save(RESULTS_DIR / "oof_mlp.npy", oof_proba)
    np.save(RESULTS_DIR / "test_mlp.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    test_pred_labels = le.inverse_transform(np.argmax(test_proba * threshold_weights, axis=1))
    out_path = write_submission(SUBMISSIONS_DIR, "mlp_v1.csv", test_ids, TARGET, test_pred_labels)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
