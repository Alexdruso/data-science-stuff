"""FT-Transformer + te13 recipe — the 4th-family probe (Day-10, Anthro #1).

The Day-8 FTT veto predates the TE-consumer era (repaired surface, no TE
inputs); the field independently reports FTT + per-value TE at CV 0.95063.
This re-run feeds the v0.7 te13 block (all-13 string EB TE, cached "u"
namespace shared with realmlp_te13) to the FTT backbone.

Design change vs train_ftt: each feature's 3-class TE posterior triple is
FUSED into its token (per-feature linear 3->D added to the value part)
instead of appended as 39 extra tokens — token count stays 1 CLS + 13, so
attention cost is unchanged. The ABSENT pad mask is dropped: a missing
value's token is its bias + the 'nan'-level TE posterior, exactly the
information channel that made the recipe pay elsewhere. Balanced softmax
(logit-adjust tau) + label smoothing kept from train_ftt.

Gate: 1-fold probe first — promote to the remaining folds only if fold-1
argmax is at/above realmlp_te13_u_s42's fold-1 (checkpoints make the run
resumable either way).

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_ftt_te13.py
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
from zoo_common import clear_ckpt, fold_impute_stats, te13_block_for_fold, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TOKEN, N_LAYERS, N_HEADS = 128, 3, 8
DROPOUT = 0.1
LR, WD, BATCH = 1e-3, 1e-5, 2048
EPOCHS, PATIENCE = 25, 5
TAU, LS_EPS = 1.075, 0.04
TE_TAG = "u"


class FTTTE(nn.Module):
    """FTT with per-feature TE-posterior fusion: token_i = value_i + W_i @ te_i."""

    def __init__(self, n_num: int, cat_dims: list[int]) -> None:
        super().__init__()
        n_feat = n_num + len(cat_dims)
        self.n_num = n_num
        self.num_w = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.num_b = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.cat_embs = nn.ModuleList([nn.Embedding(d, D_TOKEN) for d in cat_dims])
        self.te_w = nn.Parameter(torch.randn(n_feat, N_CLASSES, D_TOKEN) * 0.02)
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
        self, x_num: torch.Tensor, x_cat: torch.Tensor, x_te: torch.Tensor
    ) -> torch.Tensor:
        b = x_num.shape[0]
        val_toks = torch.cat(
            [
                x_num.unsqueeze(-1) * self.num_w + self.num_b,
                torch.stack(
                    [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1
                ),
            ],
            dim=1,
        )
        te_toks = torch.einsum("bfc,fcd->bfd", x_te, self.te_w)
        toks = torch.cat([self.cls.expand(b, -1, -1), val_toks + te_toks], dim=1)
        return self.head(self.encoder(toks)[:, 0])


def main() -> None:
    ds = load_dataset()
    assert ds.train_fit is None, "S6E7_REPAIR_FIT not wired for this trainer"
    nd = NNData(ds)
    n_num = len(nd.num_cols)
    tok_cols = nd.num_cols + nd.cat_cols  # token order
    # te13 names iterate ds.train.columns (= feature_cols); map to token order
    te_perm = [ds.feature_cols.index(c) for c in tok_cols]
    print(f"device {DEVICE}  tokens: 1 CLS + {len(tok_cols)} fused(value+TE)")

    miss_num_tr = np.isnan(nd.Xnum_tr)
    miss_num_te = np.isnan(nd.Xnum_te)

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        med, mean, std = fold_impute_stats(nd.Xnum_tr, tr_idx)
        zn_tr = ((np.where(miss_num_tr, med, nd.Xnum_tr) - mean) / std).astype(
            np.float32
        )
        zn_te = ((np.where(miss_num_te, med, nd.Xnum_te) - mean) / std).astype(
            np.float32
        )
        zn_tr[miss_num_tr] = 0.0
        zn_te[miss_num_te] = 0.0

        te_tr, te_te, _names = te13_block_for_fold(ds, tr_idx, seed, fold, TE_TAG)
        te_tr3 = te_tr.reshape(len(te_tr), -1, N_CLASSES)[:, te_perm, :]
        te_te3 = te_te.reshape(len(te_te), -1, N_CLASSES)[:, te_perm, :]

        Xn = torch.as_tensor(zn_tr, device=DEVICE)
        Xc = torch.as_tensor(nd.Xcat_tr, device=DEVICE)
        Xt = torch.as_tensor(te_tr3, device=DEVICE)
        yt = torch.as_tensor(ds.y, device=DEVICE)

        counts = np.bincount(ds.y[tr_idx], minlength=N_CLASSES).astype(np.float64)
        log_prior = np.log(counts) - np.log(counts).mean()
        adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
        crit = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

        model = FTTTE(n_num, nd.cardinalities).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        def predict(arrays) -> np.ndarray:  # noqa: ANN001
            model.eval()
            xn, xc, xt = arrays
            out = []
            with torch.no_grad():
                for s in range(0, len(xn), 8192):
                    with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                        logits = model(
                            xn[s : s + 8192], xc[s : s + 8192], xt[s : s + 8192]
                        )
                    out.append(torch.softmax(logits.float(), 1).cpu().numpy())
            return np.concatenate(out).astype(np.float64)

        val_t = torch.as_tensor(val_idx, device=DEVICE)
        tr_t = torch.as_tensor(tr_idx, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(fold_seed)
        best_ba, wait, best_state = -1.0, 0, None
        for ep in range(EPOCHS):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
            for i in range(0, len(perm), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad()
                with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    loss = crit(model(Xn[b], Xc[b], Xt[b]) + adj, yt[b])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()
            vp = predict((Xn[val_t], Xc[val_t], Xt[val_t]))
            ba = balanced_accuracy_score(ds.y[val_idx], vp.argmax(1))
            print(f"    ep {ep}: val bacc {ba:.4f}", flush=True)
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
        val_proba = predict((Xn[val_t], Xc[val_t], Xt[val_t]))
        xn_te = torch.as_tensor(zn_te, device=DEVICE)
        xc_te = torch.as_tensor(nd.Xcat_te, device=DEVICE)
        xt_te = torch.as_tensor(te_te3, device=DEVICE)
        test_proba = predict((xn_te, xc_te, xt_te))
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_ba:.4f}")

        del model, opt, sched, Xn, Xc, Xt, yt, tr_t, val_t, xn_te, xc_te, xt_te
        del best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"ftt_te13_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("ftt_te13", ds, oof, test, fold_scores)
    clear_ckpt(f"ftt_te13_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
