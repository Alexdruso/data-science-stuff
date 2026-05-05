"""Optuna hyperparameter search for PS S6E5 — MLP pit stop predictor."""

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 3
N_TRIALS = 50
EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 8192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        layer_sizes: list[int],
        dropout: float,
        use_gelu: bool,
    ) -> None:
        super().__init__()
        act_cls: type[nn.Module] = nn.GELU if use_gelu else nn.ReLU
        layers: list[nn.Module] = []
        in_size = n_features
        for size in layer_sizes:
            layers += [
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),
                act_cls(),
                nn.Dropout(dropout),
            ]
            in_size = size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_layer_sizes(n_layers: int, first_size: int, shrink: float) -> list[int]:
    sizes = [first_size]
    for _ in range(n_layers - 1):
        sizes.append(max(32, int(sizes[-1] * shrink)))
    return sizes


def prepare_arrays(
    train_pl: pl.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_pd = train_pl.to_pandas()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_pd[col])
        train_pd[col] = le.transform(train_pd[col]).astype(np.float32)

    X = train_pd[feature_cols].to_numpy(dtype=np.float32)
    y = train_pd[TARGET].to_numpy(dtype=np.float32)
    return X, y


def _train_fold_tune(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    layer_sizes: list[int],
    dropout: float,
    use_gelu: bool,
    lr: float,
    weight_decay: float,
    num_idx: list[int],
    cat_idx: list[int],
) -> float:
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    sc = StandardScaler()

    X_tr_s = np.empty_like(X_tr)
    X_val_s = np.empty_like(X_val)

    X_tr_s[:, num_idx] = pt.fit_transform(X_tr[:, num_idx])
    X_val_s[:, num_idx] = pt.transform(X_val[:, num_idx])

    X_tr_s[:, cat_idx] = sc.fit_transform(X_tr[:, cat_idx])
    X_val_s[:, cat_idx] = sc.transform(X_val[:, cat_idx])

    model = MLP(X_tr_s.shape[1], layer_sizes, dropout, use_gelu).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr_s).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_val_t = torch.from_numpy(X_val_s).to(DEVICE)

    best_val_auc = 0.0
    patience_counter = 0
    best_val_pred: np.ndarray = np.zeros(len(X_val))

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
            val_logits = model(X_val_t).cpu().numpy()
        val_pred = 1 / (1 + np.exp(-val_logits))
        val_auc = float(roc_auc_score(y_val, val_pred))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_pred = val_pred.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return float(roc_auc_score(y_val, best_val_pred))


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    num_idx: list[int],
    cat_idx: list[int],
) -> float:
    n_layers = trial.suggest_int("n_layers", 2, 4)
    first_size = trial.suggest_categorical("first_size", [128, 256, 512, 1024])
    shrink = trial.suggest_float("shrink", 0.3, 0.8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])

    layer_sizes = build_layer_sizes(n_layers, int(first_size), shrink)
    use_gelu = activation == "gelu"

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_aucs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        auc = _train_fold_tune(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            layer_sizes,
            dropout,
            use_gelu,
            lr,
            weight_decay,
            num_idx,
            cat_idx,
        )
        fold_aucs.append(auc)

    return float(np.mean(fold_aucs))


def main() -> None:
    print(f"Device: {DEVICE}")
    print("Preparing data...")

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train_pl = build_features(train_raw)
    train_pl = compute_group_features(train_raw, train_pl)

    cat_cols = [
        c
        for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train_pl.columns if c not in ("id", TARGET)]

    cat_set = set(cat_cols)
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_set]
    num_idx = [i for i, c in enumerate(feature_cols) if c not in cat_set]

    X, y = prepare_arrays(train_pl, cat_cols, feature_cols)
    print(f"X shape: {X.shape}, num_idx: {len(num_idx)}, cat_idx: {len(cat_idx)}")

    print(f"Running {N_TRIALS} Optuna trials ({N_FOLDS}-fold CV each)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, num_idx, cat_idx),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest {N_FOLDS}-fold AUC: {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "best_params_mlp.json"
    with out.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
