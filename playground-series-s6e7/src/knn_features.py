"""kNN label-distribution features in the leakage numerics (Day-10 PM, #1).

Field-verified channel (r1ck.sync 2026-07-12, honest nested solo 0.95054): the
generator injected label leakage into the non-rule columns (class-conditional
step_count etc., rule-from-irrelevant-cols AUC 0.767 vs 0.505 on the original).
The class distribution among the k nearest neighbours in the 6 leakage
numerics is a JOINT-density statistic that univariate te13 (per-value
marginals) cannot represent and 33-leaf regularized trees cannot compose.
Kill-list check: probe_linkage killed matching-to-EXTERNAL rows (different
statistic); train_dp killed label-FREE driver posteriors; te3p killed pair
TEs. Nothing covers label-derived joint-density features.

Construction (invariant #3, fold-safe): per outer fold, the neighbour INDEX =
the fold's training rows only, z-scored on fold-train stats, NaN -> 0 (mean).
Queries = all train rows + all test rows; index rows drop their self-match.
Features = neighbour label distribution at k in {10, 30, 100} -> 9 columns.
Cached per (seed, fold) like the TE blocks: results/cache/knn_u_s{seed}_f{n}.npz

Run (precompute all folds on GPU): python src/knn_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))
from train_common import SEEDS, Dataset, load_dataset
from zoo_common import CACHE_DIR

LEAK_COLS = [
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
KS = [10, 30, 100]
K_MAX = max(KS)
N_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_CHUNK, I_CHUNK = 8192, 32768

KNN_NAMES = [f"knn{k}_c{c}" for k in KS for c in range(N_CLASSES)]


def _zscore(ds: Dataset, tr_idx: NDArray[np.int64]) -> tuple[np.ndarray, np.ndarray]:
    x_tr = ds.train[LEAK_COLS].to_numpy(dtype=np.float32)
    x_te = ds.test[LEAK_COLS].to_numpy(dtype=np.float32)
    mean = np.nanmean(x_tr[tr_idx], axis=0)
    std = np.nanstd(x_tr[tr_idx], axis=0) + 1e-9
    z_tr = np.nan_to_num((x_tr - mean) / std, nan=0.0)
    z_te = np.nan_to_num((x_te - mean) / std, nan=0.0)
    return z_tr, z_te


def _topk_labels(
    queries: torch.Tensor,
    index: torch.Tensor,
    y_index: torch.Tensor,
    drop_self_of: torch.Tensor | None,
) -> torch.Tensor:
    """(n_query, K_MAX[+1]) neighbour labels by ascending distance."""
    k_eff = K_MAX + (1 if drop_self_of is not None else 0)
    out = torch.empty((len(queries), k_eff), dtype=torch.int8, device=DEVICE)
    for qs in range(0, len(queries), Q_CHUNK):
        q = queries[qs : qs + Q_CHUNK]
        best_d = torch.full((len(q), k_eff), float("inf"), device=DEVICE)
        best_y = torch.zeros((len(q), k_eff), dtype=torch.int8, device=DEVICE)
        for is_ in range(0, len(index), I_CHUNK):
            d = torch.cdist(q, index[is_ : is_ + I_CHUNK])
            k_here = min(k_eff, d.shape[1])
            dk, ik = torch.topk(d, k_here, largest=False)
            yk = y_index[is_ : is_ + I_CHUNK][ik].to(torch.int8)
            cat_d = torch.cat([best_d, dk], dim=1)
            cat_y = torch.cat([best_y, yk], dim=1)
            best_d, sel = torch.topk(cat_d, k_eff, largest=False)
            best_y = torch.gather(cat_y, 1, sel)
        out[qs : qs + Q_CHUNK] = best_y
    return out


def _dist_features(labels: torch.Tensor, skip_first: bool) -> np.ndarray:
    """Neighbour class distributions at each k (prefix means over one-hots)."""
    lab = labels[:, 1:] if skip_first else labels[:, :K_MAX]
    onehot = torch.zeros(
        (lab.shape[0], lab.shape[1], N_CLASSES), dtype=torch.float16, device=DEVICE
    )
    onehot.scatter_(2, lab.long().unsqueeze(-1), 1.0)
    cum = onehot.cumsum(dim=1)
    feats = [cum[:, k - 1, :] / k for k in KS]
    return torch.cat(feats, dim=1).float().cpu().numpy().astype(np.float32)


def knn_block_for_fold(
    ds: Dataset,
    tr_idx: NDArray[np.int64],
    seed: int,
    fold: int,
    tag: str = "u",
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"knn_{tag}_s{seed}_f{fold}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["test"], KNN_NAMES

    z_tr, z_te = _zscore(ds, tr_idx)
    index = torch.as_tensor(z_tr[tr_idx], device=DEVICE)
    y_index = torch.as_tensor(ds.y[tr_idx], device=DEVICE)

    train_block = np.empty((len(ds.y), len(KNN_NAMES)), dtype=np.float32)
    in_tr = np.zeros(len(ds.y), dtype=bool)
    in_tr[tr_idx] = True

    # index rows query their own index (self at distance 0 -> skip first)
    lab = _topk_labels(index, index, y_index, drop_self_of=index)
    train_block[tr_idx] = _dist_features(lab, skip_first=True)
    # val rows + test rows: plain top-K
    val_idx = np.where(~in_tr)[0]
    lab = _topk_labels(
        torch.as_tensor(z_tr[val_idx], device=DEVICE), index, y_index, None
    )
    train_block[val_idx] = _dist_features(lab, skip_first=False)
    lab = _topk_labels(torch.as_tensor(z_te, device=DEVICE), index, y_index, None)
    test_block = _dist_features(lab, skip_first=False)

    del index, y_index, lab
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(cache, train=train_block, test=test_block)
    return train_block, test_block, KNN_NAMES


def main() -> None:
    from sklearn.model_selection import StratifiedKFold

    ds = load_dataset()
    seed = SEEDS[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, _val_idx) in enumerate(skf.split(ds.y, ds.y), 1):
        import time

        t0 = time.time()
        tr_blk, te_blk, _ = knn_block_for_fold(ds, tr_idx, seed, fold)
        print(
            f"fold {fold}: cached ({time.time() - t0:.0f}s) "
            f"train {tr_blk.shape} test {te_blk.shape} "
            f"mean k100 dist {tr_blk[:, 6:9].mean(0).round(3)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
