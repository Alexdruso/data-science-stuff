"""Logit-adjusted (balanced-softmax) MLP for PS S6E7 — the NN done RIGHT for this metric.

The existing `train_mlp.py` trains with plain class-weighted CrossEntropyLoss. For a
severely imbalanced *balanced-accuracy* task that is the under-built choice: reweighting
rescales gradients but leaves the decision boundary a plug-in of P(y|x) under the training
prior, so the network's gains landed in non-test-like rows and did NOT transfer (adv_eval:
+MLP tied the GBDT core on the top-30% most test-like rows). Logit adjustment (Menon et al.
2020) / Balanced Softmax (Ren et al. 2020) instead adds log-prior offsets to the logits
*inside* the loss and predicts with the raw logits, which makes the trained classifier a
plug-in for the *balanced* error directly — the principled lever the MLP was missing.

Same inputs/architecture as train_mlp.py (embeddings + std numerics + missingness feats),
reused verbatim; only the loss and the (removed) class-weighting change. tau=1.0 is parameter
-free Balanced Softmax; S6E7_LA_TAU overrides. Gate it with diag_mlp_transfer.py before it can
join any ensemble (must lift adv-weighted OOF AND add test-like-region diversity).

GPU memory rule (RTX 2060, 6 GB): del model+optimizer+tensors and empty_cache() every fold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, bagged_cv, finalize, load_dataset
from train_mlp import (  # reuse architecture + data prep verbatim
    BATCH,
    DEVICE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    MLP,
    NNData,
    _predict,
    _standardize,
)

# tau=1.0 == Balanced Softmax (parameter-free); tau in (0,1] interpolates to plain softmax.
LA_TAU: float = float(os.environ.get("S6E7_LA_TAU", "1.0"))


def _log_prior(y_tr: NDArray[np.int64]) -> torch.Tensor:
    """log class frequencies of the training fold, as a (K,) tensor on DEVICE."""
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    prior = counts / counts.sum()
    return torch.as_tensor(np.log(prior), dtype=torch.float32, device=DEVICE)


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

        # Logit-adjusted loss: train on (logits + tau*log_prior), predict on raw logits.
        # No class weight — the log-prior offset is the imbalance correction.
        adj = LA_TAU * _log_prior(y[tr_idx])
        crit = nn.CrossEntropyLoss()
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
                    logits = model(cat_all[b], dense_all[b])
                    loss = crit(logits.float() + adj, yt[b])
                loss.backward()
                opt.step()
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


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}   device: {DEVICE}")
    print(f"logit-adjustment tau = {LA_TAU}")
    nd = NNData(ds)
    print(f"cat cardinalities: {dict(zip(nd.cat_cols, nd.cardinalities))}")
    oof, test, fold_scores = bagged_cv(ds, make_fit_fold(nd, ds.y))
    finalize("mlp_la", ds, oof, test, fold_scores)


if __name__ == "__main__":
    main()
