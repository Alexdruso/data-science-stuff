"""GRU sequence model for PS S6E5 — F1 Pit Stop Prediction.

Groups laps by (Driver, Race, Year) to form variable-length sequences and runs a
GRU to learn per-lap pit probabilities, capturing temporal strategy patterns that
the MLP misses (undercuts, degradation curves, strategic suppression).

CV: GroupKFold so no sequence is ever split across train/val.
Outputs: oof_rnn.npy, test_rnn.npy — same row-order convention as all other models.
Loads tuned hyperparameters from results/best_params_rnn.json when available.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import DRIVER_COLS, build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET = "PitNextLap"
N_FOLDS = 5
EPOCHS = 50
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults (overridden by best_params_rnn.json when present)
_DEFAULTS: dict[str, object] = {
    "hidden_size": 256,
    "n_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "bidirectional": False,
    "batch_size": 64,
}


def load_params() -> dict[str, object]:
    params_path = RESULTS_DIR / "best_params_rnn.json"
    base: dict[str, object] = dict(_DEFAULTS)
    if params_path.exists():
        with params_path.open() as f:
            base.update(json.load(f))
        print(f"Loaded tuned params from {params_path}", flush=True)
    return base


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            n_features,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_in = hidden_size * 2 if bidirectional else hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.rnn(x_packed)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)  # (B, T, H[*2])
        return self.head(out_padded).squeeze(-1)  # (B, T)


# ---------------------------------------------------------------------------
# Dataset / collation
# ---------------------------------------------------------------------------

class RaceSequenceDataset(Dataset):
    """One item per (Driver, Race, Year) sequence, filtered by row_mask."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray,
        row_mask: np.ndarray | None = None,
    ) -> None:
        self.seqs: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.row_indices: list[np.ndarray] = []

        if row_mask is None:
            row_mask = np.ones(len(X), dtype=bool)

        for gid in np.unique(group_ids[row_mask]):
            idx = np.where((group_ids == gid) & row_mask)[0]
            self.seqs.append(X[idx].astype(np.float32))
            self.labels.append(y[idx].astype(np.float32))
            self.row_indices.append(idx)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.seqs[i], self.labels[i], self.row_indices[i]


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
    seqs, labels, row_indices = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence([torch.from_numpy(s) for s in seqs], batch_first=True)
    labels_padded = pad_sequence(
        [torch.from_numpy(la) for la in labels],
        batch_first=True,
        padding_value=-1.0,
    )
    return seqs_padded, labels_padded, lengths, list(row_indices)


# ---------------------------------------------------------------------------
# Loss / inference helpers
# ---------------------------------------------------------------------------

def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def collect_predictions(
    model: GRUModel, loader: DataLoader, n_rows: int
) -> np.ndarray:
    """Run model on loader; return flat (n_rows,) array in original row order."""
    preds = np.zeros(n_rows, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for seqs, _, lengths, row_indices in loader:
            seqs = seqs.to(DEVICE)
            logits = model(seqs, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            for prob, row_idx, length in zip(probs, row_indices, lengths.tolist()):
                preds[row_idx] = prob[:length]
    return preds


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_arrays(
    train_pl: pl.DataFrame,
    test_pl: pl.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    num_cols = [c for c in feature_cols if c not in set(cat_cols)]

    combined_cats = pd.concat(
        [train_pd[cat_cols], test_pd[cat_cols]], ignore_index=True
    )
    ohe = pd.get_dummies(combined_cats, columns=cat_cols, dtype=np.float32)
    n_train = len(train_pd)
    train_ohe = ohe.iloc[:n_train].to_numpy(dtype=np.float32)
    test_ohe = ohe.iloc[n_train:].to_numpy(dtype=np.float32)

    X_num = train_pd[num_cols].to_numpy(dtype=np.float32)
    X_test_num = test_pd[num_cols].to_numpy(dtype=np.float32)

    X = np.hstack([X_num, train_ohe])
    X_test = np.hstack([X_test_num, test_ohe])
    y = train_pd[TARGET].to_numpy(dtype=np.float32)
    test_ids = test_pd["id"].to_numpy()

    n_num = len(num_cols)
    n_ohe = train_ohe.shape[1]
    num_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_num + n_ohe))
    return X, y, X_test, test_ids, num_idx, cat_idx


def make_group_ids(df: pl.DataFrame) -> np.ndarray:
    """Integer group ID per (Driver, Race, Year) row, in df's current row order."""
    return (
        df.select(["Driver", "Race", "Year"])
        .to_pandas()
        .groupby(["Driver", "Race", "Year"], sort=False)
        .ngroup()
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    X_scaled: np.ndarray,
    y: np.ndarray,
    X_test_scaled: np.ndarray,
    train_group_ids: np.ndarray,
    test_group_ids: np.ndarray,
    train_row_idx: np.ndarray,
    val_row_idx: np.ndarray,
    n_features: int,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
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
    test_ds = RaceSequenceDataset(
        X_test_scaled, np.zeros(len(X_test_scaled), dtype=np.float32), test_group_ids
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRUModel(n_features, hidden_size, n_layers, dropout, bidirectional).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0.0
    best_val_pred = np.zeros(len(X_scaled), dtype=np.float32)
    best_test_pred = np.zeros(len(X_test_scaled), dtype=np.float32)
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
            best_val_pred = val_preds.copy()
            best_test_pred = collect_predictions(model, test_loader, len(X_test_scaled))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    return best_val_pred, best_test_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}", flush=True)

    train_raw, test_raw = (
        pl.read_csv(DATA_DIR / "train.csv"),
        pl.read_csv(DATA_DIR / "test.csv"),
    )
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    print(f"Train: {train_pl.shape}, Test: {test_pl.shape}", flush=True)

    _exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in _exclude
    ]
    feature_cols = [c for c in train_pl.columns if c not in _exclude]

    X, y, X_test, test_ids, num_idx, cat_idx = prepare_arrays(
        train_pl, test_pl, cat_cols, feature_cols
    )
    n_features = X.shape[1]
    print(f"Features: {n_features}  ({len(num_idx)} numeric, {len(cat_idx)} OHE)", flush=True)

    train_group_ids = make_group_ids(train_pl)
    test_group_ids = make_group_ids(test_pl)
    print(f"Train sequences: {len(np.unique(train_group_ids))}", flush=True)

    params = load_params()
    print(f"Params: {params}", flush=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(X), dtype=np.float32)
    test_proba = np.zeros(len(X_test), dtype=np.float32)
    fold_aucs: list[float] = []

    for fold, (train_row_idx, val_row_idx) in enumerate(
        gkf.split(X, groups=train_group_ids), 1
    ):
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        sc = StandardScaler()
        pt.fit(X[train_row_idx][:, num_idx])
        sc.fit(X[train_row_idx][:, cat_idx])

        X_scaled = np.empty_like(X)
        X_test_scaled = np.empty_like(X_test)
        X_scaled[:, num_idx] = pt.transform(X[:, num_idx])
        X_scaled[:, cat_idx] = sc.transform(X[:, cat_idx])
        X_test_scaled[:, num_idx] = pt.transform(X_test[:, num_idx])
        X_test_scaled[:, cat_idx] = sc.transform(X_test[:, cat_idx])

        val_pred, test_pred = train_fold(
            X_scaled, y, X_test_scaled,
            train_group_ids, test_group_ids,
            train_row_idx, val_row_idx,
            n_features, params,
        )

        oof_proba[val_row_idx] = val_pred[val_row_idx]
        test_proba += test_pred / N_FOLDS

        fold_auc = float(roc_auc_score(y[val_row_idx], oof_proba[val_row_idx]))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}", flush=True)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"\nOOF AUC: {oof_auc:.4f}", flush=True)

    save_cv_result(RESULTS_DIR, "rnn_v2", fold_aucs, oof_auc)
    np.save(RESULTS_DIR / "oof_rnn.npy", oof_proba)
    np.save(RESULTS_DIR / "test_rnn.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}", flush=True)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: test_proba})
    out_path = SUBMISSIONS_DIR / "rnn_v2.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
