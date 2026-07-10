"""DANN + mask-consistency dual-loss MLP (zoo Z4, PS S6E7 — Rung 2B+C, user-pitched).

The only zoo candidate that LEARNS invariance to the train/test shift rather
than architecting around it. One backbone (train_mlp's NNData encoding), two
optional loss terms, both semi-supervised over the 295k unlabeled test rows
(label-free, leak-free):

  C — mask-consistency (Rung 2B): per batch, two independent iid Bernoulli masks
      at each column's TEST marginal NaN rate; JS-divergence penalty between the
      two views' predictions forbids routing through feature presence.
  D — gradient-reversal domain head (Rung 2C, DANN): aux head predicts
      train-vs-test from the trunk; reversed gradient (lambda ramp 2/(1+e^-10p)-1)
      strips the representation of everything domain-informative.

Training inputs are the RAW matrix (repair would hide the shift the domain head
must see). Val/test predictions are made on the REPAIRED surface (the deployed
core's surface) so the OOF is gateable by diag_mlp_transfer — hence run WITHOUT
S6E7_REPAIR but tag the artifacts _r_s42:

  S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_dann.py

Knobs: S6E7_DANN_LD (default 0.1, 0 disables D), S6E7_CONS_LC (default 1.0,
0 disables C). GPU memory rule enforced per fold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import (
    N_CLASSES,
    SEEDS,
    Dataset,
    _uniform_remask,
    finalize,
    load_dataset,
)
from train_mlp import KEY_DRIVERS, NNData
from zoo_common import clear_ckpt, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_D = float(os.environ.get("S6E7_DANN_LD", "0.1"))
LAMBDA_C = float(os.environ.get("S6E7_CONS_LC", "1.0"))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):  # noqa: ANN001, ANN205
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001, ANN205
        return -ctx.lamb * grad, None


class DannMLP(nn.Module):
    def __init__(self, cardinalities: list[int], n_dense: int) -> None:
        super().__init__()
        emb_dims = [min(4, c) for c in cardinalities]
        self.embs = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims)]
        )
        in_dim = sum(emb_dims) + n_dense
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(128, N_CLASSES)
        self.dom_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, x_cat: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.trunk(torch.cat([*e, x_dense], dim=1))

    def forward(self, x_cat: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.features(x_cat, x_dense))


def js_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(p_logits, 1)
    q = torch.softmax(q_logits, 1)
    m = (0.5 * (p + q)).clamp_min(1e-8).log()
    kl = nn.functional.kl_div
    return 0.5 * (kl(m, p, reduction="batchmean") + kl(m, q, reduction="batchmean"))


class MaskedViews:
    """Precomputed encodings + per-batch Bernoulli masking at TEST NaN rates."""

    def __init__(self, ds: Dataset, nd: NNData) -> None:
        self.nd = nd
        cols = nd.num_cols + nd.cat_cols
        self.rates = torch.tensor(
            [float(ds.test[c].isna().mean()) for c in cols],
            dtype=torch.float32,
            device=DEVICE,
        )
        self.n_num = len(nd.num_cols)
        # indices of the key drivers inside the miss block for the count feature
        self.key_idx = [cols.index(c) for c in KEY_DRIVERS]

    def apply(
        self,
        cat: torch.Tensor,
        z: torch.Tensor,
        miss: torch.Tensor,
        med_z: torch.Tensor,
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One masked view: returns (cat_view, dense_view)."""
        b = cat.shape[0]
        hit = torch.rand(b, len(self.rates), generator=gen, device=DEVICE) < self.rates
        z = torch.where(hit[:, : self.n_num], med_z, z)
        cat = torch.where(hit[:, self.n_num :], torch.zeros_like(cat), cat)
        miss = torch.maximum(miss, hit.float())
        key_count = miss[:, self.key_idx].sum(dim=1, keepdim=True)
        return cat, torch.cat([z, miss, key_count], dim=1)


def main() -> None:
    assert os.environ.get("S6E7_REPAIR", "") in ("", "0"), (
        "run WITHOUT S6E7_REPAIR: training needs the raw shift; "
        "val/test surfaces are repaired internally"
    )
    ds = load_dataset()  # raw
    train_rep = _uniform_remask(ds.train, ds.test)  # repaired surface for val preds
    nd = NNData(ds)
    ds_rep = Dataset(
        train=train_rep,
        test=ds.test,
        y=ds.y,
        test_ids=ds.test_ids,
        feature_cols=ds.feature_cols,
        cat_cols=ds.cat_cols,
    )
    nd_rep = NNData(ds_rep)
    assert nd_rep.cardinalities == nd.cardinalities  # shared vocab
    mv = MaskedViews(ds, nd)
    print(
        f"device {DEVICE}  lambda_d={LAMBDA_D} lambda_c={LAMBDA_C}  "
        f"mask rates {[round(float(r), 3) for r in mv.rates]}"
    )

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
        mean = np.where(np.isnan(nd.Xnum_tr), med, nd.Xnum_tr)[tr_idx].mean(axis=0)
        std = np.where(np.isnan(nd.Xnum_tr), med, nd.Xnum_tr)[tr_idx].std(axis=0) + 1e-6

        def enc(
            nd_x: NNData, split: str
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num = nd_x.Xnum_tr if split == "train" else nd_x.Xnum_te
            cat = nd_x.Xcat_tr if split == "train" else nd_x.Xcat_te
            miss = nd_x.miss_tr if split == "train" else nd_x.miss_te
            z = ((np.where(np.isnan(num), med, num) - mean) / std).astype(np.float32)
            return (
                torch.as_tensor(cat, device=DEVICE),
                torch.as_tensor(z, device=DEVICE),
                torch.as_tensor(miss[:, :-1].astype(np.float32), device=DEVICE),
            )

        cat_tr, z_tr, miss_tr = enc(nd, "train")  # raw: training + domain
        cat_te, z_te, miss_te = enc(nd, "test")  # raw test: unlabeled SSL
        cat_vr, z_vr, miss_vr = enc(nd_rep, "train")  # repaired: val surface
        med_z = torch.as_tensor(((med - mean) / std).astype(np.float32), device=DEVICE)
        yt = torch.as_tensor(ds.y, device=DEVICE)

        def dense_of(z: torch.Tensor, miss: torch.Tensor) -> torch.Tensor:
            key_count = miss[:, mv.key_idx].sum(dim=1, keepdim=True)
            return torch.cat([z, miss, key_count], dim=1)

        cw = compute_class_weight(
            "balanced", classes=np.arange(N_CLASSES), y=ds.y[tr_idx]
        )
        crit = nn.CrossEntropyLoss(
            weight=torch.as_tensor(cw, dtype=torch.float32, device=DEVICE)
        )
        bce = nn.BCEWithLogitsLoss()
        model = DannMLP(nd.cardinalities, z_tr.shape[1] + miss_tr.shape[1] + 1).to(
            DEVICE
        )
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        tr_t = torch.as_tensor(tr_idx, device=DEVICE)
        n_test = len(ds.test_ids)
        gen = torch.Generator(device=DEVICE).manual_seed(fold_seed)
        best_score, best_state, wait = -1.0, None, 0
        steps_per_ep = (len(tr_t) + BATCH - 1) // BATCH
        total_steps = MAX_EPOCHS * steps_per_ep
        step = 0
        for _epoch in range(MAX_EPOCHS):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
            for i in range(0, len(perm), BATCH):
                b = perm[i : i + BATCH]
                u = torch.randint(0, n_test, (len(b),), generator=gen, device=DEVICE)
                lamb = LAMBDA_D * (
                    2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                )
                step += 1
                opt.zero_grad()
                # two masked views of the labeled batch + two of the unlabeled batch
                c1, d1 = mv.apply(cat_tr[b], z_tr[b], miss_tr[b], med_z, gen)
                c2, d2 = mv.apply(cat_tr[b], z_tr[b], miss_tr[b], med_z, gen)
                u1, e1 = mv.apply(cat_te[u], z_te[u], miss_te[u], med_z, gen)
                u2, e2 = mv.apply(cat_te[u], z_te[u], miss_te[u], med_z, gen)
                f1 = model.features(c1, d1)
                logits1 = model.cls_head(f1)
                loss = crit(logits1, yt[b])
                if LAMBDA_C > 0:
                    logits2 = model(c2, d2)
                    loss = loss + LAMBDA_C * js_div(logits1, logits2)
                    loss = loss + LAMBDA_C * js_div(model(u1, e1), model(u2, e2))
                if LAMBDA_D > 0:
                    fu = model.features(u1, e1)
                    feats = torch.cat([f1, fu], dim=0)
                    dom_y = torch.cat(
                        [torch.zeros(len(b), 1), torch.ones(len(b), 1)]
                    ).to(DEVICE)
                    dom_logit = model.dom_head(GradReverse.apply(feats, lamb))
                    loss = loss + bce(dom_logit, dom_y)
                loss.backward()
                opt.step()
            # early stop on the REPAIRED val surface (deployment surface)
            model.eval()
            vi = torch.as_tensor(val_idx, device=DEVICE)
            with torch.no_grad():
                vp = _predict(model, cat_vr[vi], dense_of(z_vr[vi], miss_vr[vi]))
            score = balanced_accuracy_score(ds.y[val_idx], vp.argmax(1))
            if score > best_score:
                best_score, wait = score, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        vi = torch.as_tensor(val_idx, device=DEVICE)
        with torch.no_grad():
            val_proba = _predict(model, cat_vr[vi], dense_of(z_vr[vi], miss_vr[vi]))
            test_proba = _predict(model, cat_te, dense_of(z_te, miss_te))
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_score:.4f}")

        del model, opt, cat_tr, z_tr, miss_tr, cat_te, z_te, miss_te
        del cat_vr, z_vr, miss_vr, yt, tr_t, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"dann_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("dann", ds, oof, test, fold_scores)
    clear_ckpt(f"dann_s{SEEDS[0]}")


def _predict(
    model: nn.Module, x_cat: torch.Tensor, x_dense: torch.Tensor
) -> np.ndarray:
    out = []
    for i in range(0, len(x_cat), 8192):
        logits = model(x_cat[i : i + 8192], x_dense[i : i + 8192])
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


if __name__ == "__main__":
    main()
