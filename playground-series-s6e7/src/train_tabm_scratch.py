"""TabM from scratch (Day-9 #3): BatchEnsemble MLP + balanced softmax.

The pytabkit TabM verdict (0.9411) was CONFIG, not architecture — it has no
loss hook, so plain CE majority-walled the minorities (s6e6 lesson repeating).
This is the pure-PyTorch rebuild with the knob that made realmlp work: additive
log-prior logit adjustment IN THE TRAINING LOSS (tau=1.075; eval logits are
unadjusted, the mlp_la/ftt pattern == balanced softmax), plus the same deferred
NN input recipe (TE block via zoo_common — surface-derived cache tag, shared
with realmlp).

Architecture (TabM-mini): shared 3-layer MLP trunk; k=8 ensemble members
realized as rank-1 BatchEnsemble adapters (per-member input scale r_i, output
scale s_i, bias b_i per layer; random-sign init) + per-member linear heads.
Prediction = mean of member softmaxes. Honest EV: same family slot as realmlp
(high overlap expected); value = one more leg for the LR combiner, which
extracts whiskers from near-redundant legs (mlp_la_r precedent).

Run: S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_tabm_scratch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_mlp import NNData
from zoo_common import clear_ckpt, te_block_for_fold, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_MEMBERS = 8
HIDDEN, N_LAYERS = 512, 3
EMB_DIM = 8
DROPOUT = 0.15
LR, WD, BATCH = 2e-3, 1e-5, 1024
EPOCHS, PATIENCE = 40, 8
TAU = 1.075  # balanced-softmax strength (realmlp's loss_prior_power)
# Day-12 fix: the TE cache tag was hardcoded "r" — a u-surface run (REPAIR
# unset) would silently consume repaired-surface TE blocks. Derive it from the
# env exactly like train_realmlp_td.py.
_MULT = float(os.environ.get("S6E7_REPAIR_MULT", "1.0"))
_REPAIR = os.environ.get("S6E7_REPAIR", "") not in ("", "0")
SURFACE_TAG = ("r" if _MULT == 1.0 else "r2") if _REPAIR else "u"


class BELinear(nn.Module):
    """Shared Linear wrapped with per-member rank-1 adapters (BatchEnsemble)."""

    def __init__(self, d_in: int, d_out: int, k: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        # random-sign init (BatchEnsemble): members start diverse but unbiased
        self.r = nn.Parameter(torch.empty(k, d_in).bernoulli_(0.5) * 2 - 1)
        self.s = nn.Parameter(torch.empty(k, d_out).bernoulli_(0.5) * 2 - 1)
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, d_in) -> (B, k, d_out)
        return self.lin(x * self.r) * self.s + self.b


class TabMScratch(nn.Module):
    def __init__(self, n_cont: int, cat_dims: list[int]) -> None:
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(d, EMB_DIM) for d in cat_dims])
        d = n_cont + EMB_DIM * len(cat_dims)
        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            self.layers.append(BELinear(d if i == 0 else HIDDEN, HIDDEN, K_MEMBERS))
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Parameter(torch.randn(K_MEMBERS, HIDDEN, N_CLASSES) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(K_MEMBERS, N_CLASSES))

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        z = torch.cat([x_cont, cat], dim=1)
        z = z.unsqueeze(1).expand(-1, K_MEMBERS, -1)
        for layer in self.layers:
            z = self.drop(torch.relu(layer(z)))
        # (B, k, H) x (k, H, C) -> (B, k, C) member logits
        return torch.einsum("bkh,khc->bkc", z, self.head) + self.head_b


def fit_fold_impl(ds, nd, tr_idx, val_idx, seed, fold, want_test=True):  # noqa: ANN001, ANN201
    """One fold's fit+predict, module-level so tune_tabm_scratch.py can reuse it
    (the tuner monkey-patches the module constants per trial). want_test=False
    skips the 295k-row test pass and returns (val_proba, None)."""
    fold_seed = seed + fold * 100
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)  # noqa: NPY002

    te_tr_full, te_te, _names = te_block_for_fold(ds, tr_idx, seed, fold, SURFACE_TAG)
    num_med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
    te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
    cont_tr = np.concatenate(
        [
            np.where(np.isnan(nd.Xnum_tr), num_med, nd.Xnum_tr),
            nd.miss_tr,
            np.where(np.isnan(te_tr_full), te_med, te_tr_full),
        ],
        axis=1,
    )
    cont_te = np.concatenate(
        [
            np.where(np.isnan(nd.Xnum_te), num_med, nd.Xnum_te),
            nd.miss_te,
            np.where(np.isnan(te_te), te_med, te_te),
        ],
        axis=1,
    )
    mean = cont_tr[tr_idx].mean(axis=0)
    std = cont_tr[tr_idx].std(axis=0) + 1e-6
    cont_tr = ((cont_tr - mean) / std).astype(np.float32)
    cont_te = ((cont_te - mean) / std).astype(np.float32)

    Xn = torch.as_tensor(cont_tr, device=DEVICE)
    Xc = torch.as_tensor(nd.Xcat_tr, device=DEVICE)
    yt = torch.as_tensor(ds.y, device=DEVICE)

    counts = np.bincount(ds.y[tr_idx], minlength=N_CLASSES).astype(np.float64)
    log_prior = np.log(counts) - np.log(counts).mean()
    adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss()

    model = TabMScratch(cont_tr.shape[1], nd.cardinalities).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    def predict(xn: torch.Tensor, xc: torch.Tensor) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(xn), 8192):
                with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    logits = model(xn[s : s + 8192], xc[s : s + 8192])
                # mean of member softmaxes, eval logits UNadjusted
                out.append(
                    torch.softmax(logits.float(), dim=2).mean(dim=1).cpu().numpy()
                )
        return np.concatenate(out).astype(np.float64)

    tr_t = torch.as_tensor(tr_idx, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(fold_seed)
    best_ba, wait, best_state = -1.0, 0, None
    for _ep in range(EPOCHS):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
        for i in range(0, len(perm), BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad()
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(Xn[b], Xc[b])  # (B, k, C)
                loss = crit(
                    (logits + adj).reshape(-1, N_CLASSES),
                    yt[b].repeat_interleave(K_MEMBERS),
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        vp = predict(
            Xn[torch.as_tensor(val_idx, device=DEVICE)],
            Xc[torch.as_tensor(val_idx, device=DEVICE)],
        )
        ba = balanced_accuracy_score(ds.y[val_idx], vp.argmax(1))
        if ba > best_ba:
            best_ba, wait = ba, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_proba = predict(
        Xn[torch.as_tensor(val_idx, device=DEVICE)],
        Xc[torch.as_tensor(val_idx, device=DEVICE)],
    )
    test_proba = None
    if want_test:
        xn_te = torch.as_tensor(cont_te, device=DEVICE)
        xc_te = torch.as_tensor(nd.Xcat_te, device=DEVICE)
        test_proba = predict(xn_te, xc_te)
        del xn_te, xc_te
    print(f"  seed {seed} fold {fold} best val balanced_acc: {best_ba:.4f}")

    del model, opt, sched, Xn, Xc, yt, tr_t, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return val_proba, test_proba


def main() -> None:
    ds = load_dataset()
    nd = NNData(ds)
    print(f"device {DEVICE}  k={K_MEMBERS} members, hidden {HIDDEN}x{N_LAYERS}")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        return fit_fold_impl(ds, nd, tr_idx, val_idx, seed, fold)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"tabms_{SURFACE_TAG}_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("tabms", ds, oof, test, fold_scores)
    clear_ckpt(f"tabms_{SURFACE_TAG}_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
