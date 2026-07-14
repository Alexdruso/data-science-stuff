"""S1 — error-placement (anti-correlation) penalty leg (Day-11).

The single genuinely-NEW experiment surviving the 3-dim-channel screen
(notes/ideas_day11.md): an OBJECTIVE that attacks the campaign's *measured*
failure signature — every NN (mlp_r, dann_r, ...) dumps its diversity into the
info-deleted MCAR region (90%+ of fixes there, ~92% error-overlap). S1 forbids
that: it penalizes AGREEMENT with the frozen u-family core (lrstack_u3) ONLY on
complete-driver ∩ test-like rows — where information EXISTS — pushing the model
to decorrelate where decorrelation could actually transfer.

Design is deliberately a minimal-diff from train_dann.py: SAME MLP backbone
(so the diag signature is directly comparable to dann_r), only the objective
changes (grad-reversal/consistency → error-placement). Unrepaired ("u") surface
throughout to match the u-family core and stack. The penalty target is the core's
cross-fitted OOF (leak-free) and the subset mask is label-free (driver-missingness
+ adversarial top-30%), so the whole objective is leak-free.

Pre-registered read (advisor): success = diag fix-share rises AND
%fixes-missing-driver FALLS below the universal ~90% for the first time, then a
12-cell LR-seat test. Kill if the signature stays ≥90% missing-driver (S1 has
reproduced the cascade/DANN outcome — error placement is a hard invariant).

Run (λ sweep, unrepaired surface):
  S6E7_S1_LAMBDA=0.3 S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_s1.py
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
from train_common import N_CLASSES, RESULTS_DIR, SEEDS, finalize, load_dataset
from train_mlp import KEY_DRIVERS, NNData
from zoo_common import clear_ckpt, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1024
MAX_EPOCHS = 60
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA = float(os.environ.get("S6E7_S1_LAMBDA", "0.3"))
CORE_KEY = "lrstack_u3_r"  # frozen u-family core (the "_r" is the raw-logits tag)


class S1MLP(nn.Module):
    """train_dann.py's DannMLP backbone, minus the domain head."""

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

    def forward(self, x_cat: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.cls_head(self.trunk(torch.cat([*e, x_dense], dim=1)))


def main() -> None:
    assert os.environ.get("S6E7_REPAIR", "") in ("", "0"), (
        "S1 runs on the UNREPAIRED (u) surface to match the u-family core/stack"
    )
    ds = load_dataset()  # raw = the "u" surface
    nd = NNData(ds)

    # frozen core (leak-free cross-fitted OOF) + the complete∩test-like subset mask
    core_np = np.load(RESULTS_DIR / f"oof_{CORE_KEY}.npy").astype(np.float32)
    core_p = torch.as_tensor(core_np, device=DEVICE)
    missing = np.zeros(len(ds.y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    testlike = adv >= np.quantile(adv, 0.70)
    s1_np = ((~missing) & testlike).astype(np.float32)
    s1_mask = torch.as_tensor(s1_np, device=DEVICE)
    print(
        f"device {DEVICE}  lambda={LAMBDA}  core={CORE_KEY}  "
        f"penalty subset {s1_np.mean():.1%} of train ({int(s1_np.sum()):,} rows)"
    )

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
        mean = np.where(np.isnan(nd.Xnum_tr), med, nd.Xnum_tr)[tr_idx].mean(axis=0)
        std = np.where(np.isnan(nd.Xnum_tr), med, nd.Xnum_tr)[tr_idx].std(axis=0) + 1e-6

        def enc(split: str) -> tuple[torch.Tensor, torch.Tensor]:
            num = nd.Xnum_tr if split == "train" else nd.Xnum_te
            cat = nd.Xcat_tr if split == "train" else nd.Xcat_te
            miss = nd.miss_tr if split == "train" else nd.miss_te
            z = ((np.where(np.isnan(num), med, num) - mean) / std).astype(np.float32)
            m = torch.as_tensor(miss[:, :-1].astype(np.float32), device=DEVICE)
            key_count = m[:, [nd.num_cols.__add__(nd.cat_cols).index(c)
                              for c in KEY_DRIVERS]].sum(dim=1, keepdim=True)
            dense = torch.cat([torch.as_tensor(z, device=DEVICE), m, key_count], dim=1)
            return torch.as_tensor(cat, device=DEVICE), dense

        cat_tr, dense_tr = enc("train")
        cat_te, dense_te = enc("test")
        yt = torch.as_tensor(ds.y, device=DEVICE)

        cw = compute_class_weight(
            "balanced", classes=np.arange(N_CLASSES), y=ds.y[tr_idx]
        )
        crit = nn.CrossEntropyLoss(
            weight=torch.as_tensor(cw, dtype=torch.float32, device=DEVICE)
        )
        model = S1MLP(nd.cardinalities, dense_tr.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        tr_t = torch.as_tensor(tr_idx, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(fold_seed)
        best_score, best_state, wait = -1.0, None, 0
        for _epoch in range(MAX_EPOCHS):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
            for i in range(0, len(perm), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad()
                logits = model(cat_tr[b], dense_tr[b])
                loss = crit(logits, yt[b])
                if LAMBDA > 0:
                    sel = s1_mask[b] > 0
                    if sel.any():
                        p = torch.softmax(logits[sel], dim=1)
                        # agreement with the frozen core on complete∩test-like rows
                        agree = (p * core_p[b][sel]).sum(dim=1).mean()
                        loss = loss + LAMBDA * agree
                loss.backward()
                opt.step()
            model.eval()
            vi = torch.as_tensor(val_idx, device=DEVICE)
            with torch.no_grad():
                vp = _predict(model, cat_tr[vi], dense_tr[vi])
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
            val_proba = _predict(model, cat_tr[vi], dense_tr[vi])
            test_proba = _predict(model, cat_te, dense_te)
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_score:.4f}")

        del model, opt, cat_tr, dense_tr, cat_te, dense_te, yt, tr_t, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    name = f"s1_l{int(round(LAMBDA * 100)):03d}"
    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"{name}_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize(name, ds, oof, test, fold_scores)
    clear_ckpt(f"{name}_s{SEEDS[0]}")


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
