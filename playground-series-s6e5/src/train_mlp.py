"""PyTorch MLP 5-fold CV training for PS S6E5 — F1 Pit Stop Prediction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5
BATCH_SIZE = 8192
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_csv(DATA_DIR / "train.csv"), pl.read_csv(DATA_DIR / "test.csv")


def prepare_arrays(
    train_pl: pl.DataFrame,
    test_pl: pl.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    # Label-encode categoricals (consistent across train+test)
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_pd[col], test_pd[col]], ignore_index=True)
        le.fit(combined)
        train_pd[col] = le.transform(train_pd[col]).astype(np.float32)
        test_pd[col] = le.transform(test_pd[col]).astype(np.float32)

    X = train_pd[feature_cols].to_numpy(dtype=np.float32)
    y = train_pd[TARGET].to_numpy(dtype=np.float32)
    X_test = test_pd[feature_cols].to_numpy(dtype=np.float32)
    test_ids = test_pd["id"].to_numpy()
    return X, y, X_test, test_ids


def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = MLP(X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    X_test_t = torch.from_numpy(X_test_scaled).to(DEVICE)

    best_val_auc = 0.0
    best_val_pred: np.ndarray = np.zeros(len(X_val))
    best_test_pred: np.ndarray = np.zeros(len(X_test))
    patience_counter = 0

    n = len(X_tr_t)
    for epoch in range(EPOCHS):
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
            with torch.no_grad():
                test_logits = model(X_test_t).cpu().numpy()
            best_test_pred = 1 / (1 + np.exp(-test_logits))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val_pred, best_test_pred


def main() -> None:
    print(f"Device: {DEVICE}")
    train_raw, test_raw = load_data()
    train_pl = build_features(train_raw)
    test_pl = build_features(test_raw)
    train_pl = compute_group_features(train_raw, train_pl)
    test_pl = compute_group_features(train_raw, test_pl)
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}")

    cat_cols = [
        c
        for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train_pl.columns if c not in ("id", TARGET)]

    X, y, X_test, test_ids = prepare_arrays(train_pl, test_pl, cat_cols, feature_cols)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        val_pred, test_pred = train_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], X_test
        )
        oof_proba[val_idx] = val_pred
        test_proba += test_pred / N_FOLDS

        fold_auc = float(roc_auc_score(y[val_idx], val_pred))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}")

    save_cv_result(RESULTS_DIR, "mlp_v1", fold_aucs, oof_auc)

    np.save(RESULTS_DIR / "oof_mlp.npy", oof_proba)
    np.save(RESULTS_DIR / "test_mlp.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "mlp_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
