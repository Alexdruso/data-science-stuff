"""FT-Transformer with ABSENT missing tokens (zoo Z2, PS S6E7) — setenc-lite.

Port of s6e6 train_ftt_deotte.py with the Rung-2A structural prior: a missing
feature's token is EXCLUDED from attention via src_key_padding_mask (true
absence — no NaN sentinel, no indicator channel, no learned "missing" vector).
The model cannot dedicate a representation to "water is missing", so the
train-only NaN<->trigger couplings that survive the R2a repair (the repair adds
NaNs but can't unmask trigger NaNs) have no direct input channel. Attention over
feature tokens is also a different inductive bias from MLP/GBDT splits.

Mechanics: numerics are z-scored on train-fold stats then missing entries are
0-filled (values never read — masked out at every encoder layer); cats use
integer codes (0 = missing, masked). Only the always-present CLS token is read.
fp16 autocast + GradScaler (Turing tensor cores).

Run: S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_ftt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, SEEDS, finalize, load_dataset
from train_mlp import NNData
from zoo_common import clear_ckpt, fold_impute_stats, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TOKEN, N_LAYERS, N_HEADS = 128, 3, 8
DROPOUT = 0.1
LR, WD, BATCH = 1e-3, 1e-5, 1024
EPOCHS, PATIENCE = 40, 8
TAU, LS_EPS = 1.075, 0.04  # logit-adjust in the train loss + label smoothing


class FTTAbsent(nn.Module):
    def __init__(self, n_num: int, cat_dims: list[int]) -> None:
        super().__init__()
        self.n_num = n_num
        self.num_w = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.num_b = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.cat_embs = nn.ModuleList([nn.Embedding(d, D_TOKEN) for d in cat_dims])
        self.cls = nn.Parameter(torch.randn(1, 1, D_TOKEN) * 0.02)
        layer = nn.TransformerEncoderLayer(
            D_TOKEN,
            N_HEADS,
            dim_feedforward=D_TOKEN * 2,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(D_TOKEN), nn.GELU(), nn.Linear(D_TOKEN, N_CLASSES)
        )

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        b = x_num.shape[0]
        toks = [
            self.cls.expand(b, -1, -1),
            x_num.unsqueeze(-1) * self.num_w + self.num_b,
            torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1
            ),
        ]
        # pad: (b, 1 + n_num + n_cat) bool, True = missing token, excluded from
        # attention at EVERY layer; CLS (col 0) is never masked and is all we read.
        z = self.encoder(torch.cat(toks, dim=1), src_key_padding_mask=pad)
        return self.head(z[:, 0])


def main() -> None:
    ds = load_dataset()
    nd = NNData(ds)
    n_num = len(nd.num_cols)
    print(f"device {DEVICE}  tokens: 1 CLS + {n_num} num + {len(nd.cat_cols)} cat")

    miss_num_tr = np.isnan(nd.Xnum_tr)
    miss_num_te = np.isnan(nd.Xnum_te)
    pad_tr = np.concatenate(
        [np.zeros((len(ds.y), 1), bool), miss_num_tr, nd.Xcat_tr == 0], axis=1
    )
    pad_te = np.concatenate(
        [np.zeros((len(ds.test_ids), 1), bool), miss_num_te, nd.Xcat_te == 0], axis=1
    )

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        med, mean, std = fold_impute_stats(nd.Xnum_tr, tr_idx)
        # z-score then ZERO the missing entries — the values are masked out of
        # attention, the 0 only prevents NaN propagation through projections.
        zn_tr = ((np.where(miss_num_tr, med, nd.Xnum_tr) - mean) / std).astype(
            np.float32
        )
        zn_te = ((np.where(miss_num_te, med, nd.Xnum_te) - mean) / std).astype(
            np.float32
        )
        zn_tr[miss_num_tr] = 0.0
        zn_te[miss_num_te] = 0.0

        Xn = torch.as_tensor(zn_tr, device=DEVICE)
        Xc = torch.as_tensor(nd.Xcat_tr, device=DEVICE)
        Pd = torch.as_tensor(pad_tr, device=DEVICE)
        yt = torch.as_tensor(ds.y, device=DEVICE)

        counts = np.bincount(ds.y[tr_idx], minlength=N_CLASSES).astype(np.float64)
        log_prior = np.log(counts) - np.log(counts).mean()
        adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
        crit = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

        model = FTTAbsent(n_num, nd.cardinalities).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        def predict(idx_or_arrays) -> np.ndarray:  # noqa: ANN001
            model.eval()
            if isinstance(idx_or_arrays, np.ndarray):
                xn = Xn[torch.as_tensor(idx_or_arrays, device=DEVICE)]
                xc = Xc[torch.as_tensor(idx_or_arrays, device=DEVICE)]
                pd_ = Pd[torch.as_tensor(idx_or_arrays, device=DEVICE)]
            else:
                xn, xc, pd_ = idx_or_arrays
            out = []
            with torch.no_grad():
                for s in range(0, len(xn), 8192):
                    with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                        logits = model(
                            xn[s : s + 8192], xc[s : s + 8192], pd_[s : s + 8192]
                        )
                    out.append(torch.softmax(logits.float(), 1).cpu().numpy())
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
                    loss = crit(model(Xn[b], Xc[b], Pd[b]) + adj, yt[b])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()
            vp = predict(val_idx)
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
        val_proba = predict(val_idx)
        xn_te = torch.as_tensor(zn_te, device=DEVICE)
        xc_te = torch.as_tensor(nd.Xcat_te, device=DEVICE)
        pd_te = torch.as_tensor(pad_te, device=DEVICE)
        test_proba = predict((xn_te, xc_te, pd_te))
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_ba:.4f}")

        del model, opt, sched, Xn, Xc, Pd, yt, tr_t, xn_te, xc_te, pd_te, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"ftt_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("ftt", ds, oof, test, fold_scores)
    clear_ckpt(f"ftt_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
