"""Optuna hyperparameter tuning for the GRU sequence model (PS S6E5).

Searches over hidden_size, n_layers, dropout, lr, weight_decay, bidirectional,
batch_size using 3-fold GroupKFold (respects sequence integrity).

Best params saved to results/best_params_rnn.json for use by train_rnn.py.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from features import DRIVER_COLS, build_features, compute_group_features
from train_rnn import (
    GRUModel,
    RaceSequenceDataset,
    collate_fn,
    collect_predictions,
    make_group_ids,
    masked_bce_loss,
    prepare_arrays,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 3       # 3-fold for speed; full 5-fold happens in train_rnn.py
N_TRIALS = 30
EPOCHS = 40       # slightly fewer than train_rnn.py to keep tuning fast
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Single-fold training for tuning (returns val AUC only)
# ---------------------------------------------------------------------------

def _train_fold_tune(
    X_scaled: np.ndarray,
    y: np.ndarray,
    train_row_idx: np.ndarray,
    val_row_idx: np.ndarray,
    train_group_ids: np.ndarray,
    n_features: int,
    params: dict[str, object],
) -> float:
    batch_size = int(params["batch_size"])
    hidden_size = int(params["hidden_size"])
    n_layers = int(params["n_layers"])
    dropout = float(params["dropout"])
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])
    bidirectional = bool(params["bidirectional"])

    train_mask = np.zeros(len(X_scaled), dtype=bool)
    train_mask[train_row_idx] = True
    val_mask = np.zeros(len(X_scaled), dtype=bool)
    val_mask[val_row_idx] = True

    train_ds = RaceSequenceDataset(X_scaled, y, train_group_ids, row_mask=train_mask)
    val_ds = RaceSequenceDataset(X_scaled, y, train_group_ids, row_mask=val_mask)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRUModel(n_features, hidden_size, n_layers, dropout, bidirectional).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0.0
    patience_counter = 0

    for _epoch in range(EPOCHS):
        model.train()
        for seqs, labels, lengths, _ in train_loader:
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = masked_bce_loss(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_preds = collect_predictions(model, val_loader, len(X_scaled))
        val_auc = float(roc_auc_score(y[val_row_idx], val_preds[val_row_idx]))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return best_val_auc


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    train_group_ids: np.ndarray,
    num_idx: list[int],
    cat_idx: list[int],
    n_features: int,
) -> float:
    params: dict[str, object] = {
        "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_aucs: list[float] = []

    for fold_idx, (train_row_idx, val_row_idx) in enumerate(
        gkf.split(X, groups=train_group_ids)
    ):
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        sc = StandardScaler()
        pt.fit(X[train_row_idx][:, num_idx])
        sc.fit(X[train_row_idx][:, cat_idx])

        X_scaled = np.empty_like(X)
        X_scaled[:, num_idx] = pt.transform(X[:, num_idx])
        X_scaled[:, cat_idx] = sc.transform(X[:, cat_idx])

        auc = _train_fold_tune(
            X_scaled, y, train_row_idx, val_row_idx,
            train_group_ids, n_features, params,
        )
        fold_aucs.append(auc)

        # Prune unpromising trials after first fold
        intermediate = float(np.mean(fold_aucs))
        trial.report(intermediate, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return float(np.mean(fold_aucs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}", flush=True)

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    print(f"Train: {train_pl.shape}", flush=True)

    _exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in _exclude
    ]
    feature_cols = [c for c in train_pl.columns if c not in _exclude]

    X, y, X_test, _, num_idx, cat_idx = prepare_arrays(
        train_pl, test_pl, cat_cols, feature_cols
    )
    n_features = X.shape[1]
    train_group_ids = make_group_ids(train_pl)
    print(f"Features: {n_features}, Train sequences: {len(np.unique(train_group_ids))}", flush=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, X, y, train_group_ids, num_idx, cat_idx, n_features),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\nBest trial: {best.number}  val AUC: {best.value:.4f}", flush=True)
    print("Best params:", flush=True)
    for k, v in best.params.items():
        print(f"  {k}: {v}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    params_path = RESULTS_DIR / "best_params_rnn.json"
    with params_path.open("w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Best params saved → {params_path}", flush=True)


if __name__ == "__main__":
    main()
