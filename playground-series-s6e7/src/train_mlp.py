"""PyTorch MLP for PS S6E7 — a structurally different base for ensemble diversity.

The 3 GBDTs are at the data's information ceiling and only fix ~5-7% of each other's
errors; a non-tree model is the largest remaining lever (agents' unanimous call). This MLP
uses categorical embeddings + standardized numerics + explicit missingness features (trees
get missingness free via NaN splits; an MLP needs it spelled out — this also folds in the
Tier-B "missingness on the 3 key drivers" idea). Trains via the shared bagged 5-fold CV and
reports through finalize() so its OOF/test line up with the GBDTs for ensembling.

GPU memory rule (RTX 2060, 6 GB): del model+optimizer+tensors and empty_cache() every fold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, Dataset, bagged_cv, finalize, load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4


class NNData:
    """Precomputed, fold-independent NN inputs (shared integer codes + raw numerics + masks)."""

    def __init__(self, ds: Dataset) -> None:
        self.num_cols = [c for c in ds.feature_cols if c not in ds.cat_cols]
        self.cat_cols = ds.cat_cols

        # Categorical integer codes with a shared vocab; index 0 reserved for missing/unknown.
        self.cardinalities: list[int] = []
        cat_tr, cat_te = [], []
        for c in self.cat_cols:
            cats = sorted(set(ds.train[c].dropna()) | set(ds.test[c].dropna()))
            m = {v: i + 1 for i, v in enumerate(cats)}
            self.cardinalities.append(len(cats) + 1)
            cat_tr.append(ds.train[c].map(m).fillna(0).to_numpy(dtype=np.int64))
            cat_te.append(ds.test[c].map(m).fillna(0).to_numpy(dtype=np.int64))
        self.Xcat_tr = np.stack(cat_tr, axis=1)
        self.Xcat_te = np.stack(cat_te, axis=1)

        # Numerics (raw, NaN kept) + missingness masks for every feature + key-driver count.
        self.Xnum_tr = ds.train[self.num_cols].to_numpy(dtype=np.float64)
        self.Xnum_te = ds.test[self.num_cols].to_numpy(dtype=np.float64)
        self.miss_tr = self._missing_block(ds.train)
        self.miss_te = self._missing_block(ds.test)

    def _missing_block(self, df: object) -> NDArray[np.float64]:
        cols = self.num_cols + self.cat_cols
        assert hasattr(df, "isna")
        miss = df[cols].isna().to_numpy(dtype=np.float64)  # type: ignore[attr-defined]
        n_key = df[KEY_DRIVERS].isna().sum(axis=1).to_numpy(dtype=np.float64)[:, None]  # type: ignore[attr-defined]
        return np.concatenate([miss, n_key], axis=1)


class MLP(nn.Module):
    def __init__(self, cardinalities: list[int], n_dense: int) -> None:
        super().__init__()
        emb_dims = [min(4, c) for c in cardinalities]
        self.embs = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims)]
        )
        in_dim = sum(emb_dims) + n_dense
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x_cat: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([*e, x_dense], dim=1))


def _standardize(
    num_raw: NDArray[np.float64], tr_idx: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-fold median-impute + z-score using TRAIN-fold statistics only."""
    med = np.nanmedian(num_raw[tr_idx], axis=0)
    filled = np.where(np.isnan(num_raw), med, num_raw)
    mean = filled[tr_idx].mean(axis=0)
    std = filled[tr_idx].std(axis=0) + 1e-6
    return med, mean, std


def make_fit_fold(nd: NNData, y: NDArray[np.int64]):  # noqa: ANN201 - closure
    def fit_fold(
        tr_idx: NDArray[np.int64], val_idx: NDArray[np.int64], seed: int, fold: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        torch.manual_seed(seed)
        med, mean, std = _standardize(nd.Xnum_tr, tr_idx)

        def dense(
            num_raw: NDArray[np.float64], miss: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            z = (np.where(np.isnan(num_raw), med, num_raw) - mean) / std
            return np.concatenate([z, miss], axis=1).astype(np.float32)

        dense_tr = dense(nd.Xnum_tr, nd.miss_tr)
        cat_all = torch.as_tensor(nd.Xcat_tr, device=DEVICE)
        dense_all = torch.as_tensor(dense_tr, device=DEVICE)
        yt = torch.as_tensor(y, device=DEVICE)

        cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y[tr_idx])
        crit = nn.CrossEntropyLoss(
            weight=torch.as_tensor(cw, dtype=torch.float32, device=DEVICE)
        )
        model = MLP(nd.cardinalities, dense_tr.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        tr_t = torch.as_tensor(tr_idx, device=DEVICE)
        best_score, best_state, patience = -1.0, None, 0
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        for _epoch in range(MAX_EPOCHS):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
            for i in range(0, len(perm), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad()
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"
                ):
                    loss = crit(model(cat_all[b], dense_all[b]), yt[b])
                loss.backward()
                opt.step()
            # early stop on val balanced accuracy (argmax)
            model.eval()
            with torch.no_grad():
                vp = _predict(
                    model,
                    cat_all[torch.as_tensor(val_idx, device=DEVICE)],
                    dense_all[torch.as_tensor(val_idx, device=DEVICE)],
                )
            score = balanced_accuracy_score(y[val_idx], vp.argmax(axis=1))
            if score > best_score:
                best_score, patience = score, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_proba = _predict(
                model,
                cat_all[torch.as_tensor(val_idx, device=DEVICE)],
                dense_all[torch.as_tensor(val_idx, device=DEVICE)],
            )
            cat_te = torch.as_tensor(nd.Xcat_te, device=DEVICE)
            dense_te = torch.as_tensor(dense(nd.Xnum_te, nd.miss_te), device=DEVICE)
            test_proba = _predict(model, cat_te, dense_te)
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_score:.4f}")

        del model, opt, cat_all, dense_all, yt, tr_t, cat_te, dense_te, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    return fit_fold


def _predict(
    model: nn.Module, x_cat: torch.Tensor, x_dense: torch.Tensor
) -> NDArray[np.float64]:
    """Batched softmax probabilities, returned on CPU as (n, K) float64."""
    out = []
    for i in range(0, len(x_cat), 8192):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"
        ):
            logits = model(x_cat[i : i + 8192], x_dense[i : i + 8192])
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}   device: {DEVICE}")
    nd = NNData(ds)
    print(f"cat cardinalities: {dict(zip(nd.cat_cols, nd.cardinalities))}")
    oof, test, fold_scores = bagged_cv(ds, make_fit_fold(nd, ds.y))
    finalize("mlp", ds, oof, test, fold_scores)


if __name__ == "__main__":
    main()
