"""PyTorch MLP 5-fold CV training for PS S6E6 — Predicting Stellar Class.

3-class classifier (GALAXY/QSO/STAR) — the primary non-GBDT diversity bet for
the ensemble. Adapted from the S6E5 binary MLP template, converted to multiclass:
3-unit output + CrossEntropyLoss with inverse-frequency class weights (mirrors the
GBDTs' class_weight="balanced"), early-stopping on validation balanced accuracy,
softmax → (n_train × 3) / (n_test × 3) probability arrays for blending.

GPU-memory rule (CLAUDE.md): every fold dels its model/optimizer/tensors and the
outer loop calls torch.cuda.empty_cache() after each fold.

Run with:  python src/train_mlp.py
Auto-loads results/best_params_mlp.json if present (from tune_mlp.py).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, load_params, write_submission

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

def build_layer_sizes(params: dict[str, object]) -> list[int]:
    n = int(params["n_layers"])  # type: ignore[arg-type]
    first = int(params["first_size"])  # type: ignore[arg-type]
    shrink = float(params.get("shrink", 0.5))  # type: ignore[arg-type]
    sizes = [first]
    for _ in range(n - 1):
        sizes.append(max(32, int(sizes[-1] * shrink)))
    return sizes


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        layer_sizes: list[int],
        dropout: float,
        use_gelu: bool,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        act_cls: type[nn.Module] = nn.GELU if use_gelu else nn.ReLU
        blocks = []
        in_size = n_features
        for size in layer_sizes:
            blocks += [
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),
                act_cls(),
                nn.Dropout(dropout),
            ]
            in_size = size
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(in_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.body(x))


def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, sklearn 'balanced' formula: n / (k * count)."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    return (len(y) / (N_CLASSES * counts)).astype(np.float32)


def prepare_arrays(
    train_pl: pl.DataFrame,
    test_pl: pl.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """OHE categoricals; return (X, X_test, num_idx, cat_idx) as float32 matrices."""
    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]

    combined = pd.concat([train_pd[cat_cols], test_pd[cat_cols]], ignore_index=True)
    ohe = pd.get_dummies(combined, columns=cat_cols, dtype=np.float32)
    n_train = len(train_pd)
    train_ohe = ohe.iloc[:n_train].to_numpy(dtype=np.float32)
    test_ohe = ohe.iloc[n_train:].to_numpy(dtype=np.float32)

    X_num = train_pd[num_cols].to_numpy(dtype=np.float32)
    X_test_num = test_pd[num_cols].to_numpy(dtype=np.float32)

    X = np.hstack([X_num, train_ohe])
    X_test = np.hstack([X_test_num, test_ohe])
    n_num = len(num_cols)
    num_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_num + train_ohe.shape[1]))
    return X, X_test, num_idx, cat_idx


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
    # Per-fold scaling fit on the train split only (no leakage): Yeo-Johnson on
    # numeric cols, StandardScaler on OHE binary cols.
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    sc = StandardScaler()
    X_tr_s, X_val_s, X_test_s = (np.empty_like(a) for a in (X_tr, X_val, X_test))
    X_tr_s[:, num_idx] = pt.fit_transform(X_tr[:, num_idx])
    X_val_s[:, num_idx] = pt.transform(X_val[:, num_idx])
    X_test_s[:, num_idx] = pt.transform(X_test[:, num_idx])
    X_tr_s[:, cat_idx] = sc.fit_transform(X_tr[:, cat_idx])
    X_val_s[:, cat_idx] = sc.transform(X_val[:, cat_idx])
    X_test_s[:, cat_idx] = sc.transform(X_test[:, cat_idx])

    layer_sizes = build_layer_sizes(params)
    model = MLP(
        X_tr_s.shape[1],
        layer_sizes,
        float(params["dropout"]),  # type: ignore[arg-type]
        params["activation"] == "gelu",
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),  # type: ignore[arg-type]
        weight_decay=float(params["weight_decay"]),  # type: ignore[arg-type]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    weight_t = torch.from_numpy(class_weights(y_tr)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_t)

    X_tr_t = torch.from_numpy(X_tr_s).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
    X_val_t = torch.from_numpy(X_val_s).to(DEVICE)
    X_test_t = torch.from_numpy(X_test_s).to(DEVICE)

    best_ba = -1.0
    best_val_pred = np.zeros((len(X_val_s), N_CLASSES))
    best_test_pred = np.zeros((len(X_test_s), N_CLASSES))
    patience_counter = 0
    n = len(X_tr_t)

    for _epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_proba = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
        ba = float(balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1)))
        if ba > best_ba:
            best_ba = ba
            best_val_pred = val_proba
            with torch.no_grad():
                best_test_pred = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    del model, optimizer, scheduler, criterion, weight_t
    del X_tr_t, y_tr_t, X_val_t, X_test_t
    return best_val_pred, best_test_pred


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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        val_pred, test_pred = train_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X_test, num_idx, cat_idx, params,
        )
        oof_proba[val_idx] = val_pred
        test_proba += test_pred / N_FOLDS
        score = float(balanced_accuracy_score(y[val_idx], np.argmax(val_pred, axis=1)))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

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
