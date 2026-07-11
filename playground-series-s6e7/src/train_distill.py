"""DL-2: transductive distillation from the deployed stack on REAL TEST rows.

The Day-9 saturation result: no architecture earns an LR seat — every leg
trains on the same (repaired) train surface and lands the same errors. The one
untried mechanism class is a leg trained on DIFFERENT DATA: the 295k real test
rows are the only inputs with the TRUE mask mechanism (the repair only
emulates it). Channel = surface, not diversity (alchemist #2).

Protocol per fold (zoo_cv, m100 surface):
  A) distill: student = TabMScratch backbone; soft cross-entropy to the
     deployed final-#2 engine's test probabilities (teacher =
     test_lrstack6b_r.npy, raw meta probs) on TEST ROWS ONLY — label-free.
  B) fine-tune: fold-train labeled rows, balanced softmax (tabms recipe).
  C) predict val + test.

⚠️ Honesty flag: the teacher's test probs come from full-CV models, so val
labels influence them diffusely -> student OOF carries a small optimistic
bias. Judge by (1) paired read vs tabms_r_s42 (same backbone, no distill —
isolates the channel) with sub-0.001 positives DISCOUNTED, and (2) the diag
signature: does training on true-mask inputs move the fix placement off the
universal missing-driver concentration?

Run: S6E7_REPAIR=1 S6E7_SEEDS=42 S6E7_RUN_TAG=_r_s42 python src/train_distill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, RESULTS_DIR, SEEDS, finalize, load_dataset
from train_mlp import NNData
from train_tabm_scratch import BATCH, EPOCHS, K_MEMBERS, LR, PATIENCE, TAU, WD, TabMScratch
from zoo_common import clear_ckpt, te_block_for_fold, zoo_cv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER = "lrstack6b_r"
DISTILL_EPOCHS = 6
DISTILL_BATCH = 2048


def main() -> None:
    ds = load_dataset()
    nd = NNData(ds)
    teacher = np.load(RESULTS_DIR / f"test_{TEACHER}.npy")
    print(f"device {DEVICE}  teacher={TEACHER}  distill {DISTILL_EPOCHS}ep on "
          f"{len(teacher):,} test rows")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        fold_seed = seed + fold * 100
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)  # noqa: NPY002

        te_tr_full, te_te, _names = te_block_for_fold(ds, tr_idx, seed, fold, "r")
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
        Xn_te = torch.as_tensor(cont_te, device=DEVICE)
        Xc_te = torch.as_tensor(nd.Xcat_te, device=DEVICE)
        yt = torch.as_tensor(ds.y, device=DEVICE)
        T = torch.as_tensor(teacher.astype(np.float32), device=DEVICE)

        model = TabMScratch(cont_tr.shape[1], nd.cardinalities).to(DEVICE)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        gen = torch.Generator(device=DEVICE).manual_seed(fold_seed)

        # ---- phase A: distill on TEST rows (true mask mechanism, label-free)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, fused=True)
        n_te = len(cont_te)
        for _ep in range(DISTILL_EPOCHS):
            model.train()
            perm = torch.randperm(n_te, generator=gen, device=DEVICE)
            for i in range(0, n_te, DISTILL_BATCH):
                b = perm[i : i + DISTILL_BATCH]
                opt.zero_grad()
                with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    logits = model(Xn_te[b], Xc_te[b])  # (B, k, C)
                    logp = torch.log_softmax(logits.float(), dim=2)
                    loss = -(T[b].unsqueeze(1) * logp).sum(dim=2).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        # ---- phase B: fine-tune on fold-train labels (tabms recipe)
        counts = np.bincount(ds.y[tr_idx], minlength=N_CLASSES).astype(np.float64)
        log_prior = np.log(counts) - np.log(counts).mean()
        adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(
            model.parameters(), lr=LR / 2, weight_decay=WD, fused=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        def predict(xn: torch.Tensor, xc: torch.Tensor) -> np.ndarray:
            model.eval()
            out = []
            with torch.no_grad():
                for s in range(0, len(xn), 8192):
                    with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                        logits = model(xn[s : s + 8192], xc[s : s + 8192])
                    out.append(
                        torch.softmax(logits.float(), dim=2).mean(dim=1).cpu().numpy()
                    )
            return np.concatenate(out).astype(np.float64)

        tr_t = torch.as_tensor(tr_idx, device=DEVICE)
        val_t = torch.as_tensor(val_idx, device=DEVICE)
        best_ba, wait, best_state = -1.0, 0, None
        for _ep in range(EPOCHS):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=DEVICE)]
            for i in range(0, len(perm), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad()
                with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    logits = model(Xn[b], Xc[b])
                    loss = crit(
                        (logits + adj).reshape(-1, N_CLASSES),
                        yt[b].repeat_interleave(K_MEMBERS),
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()
            vp = predict(Xn[val_t], Xc[val_t])
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
        val_proba = predict(Xn[val_t], Xc[val_t])
        test_proba = predict(Xn_te, Xc_te)
        print(f"  seed {seed} fold {fold} best val balanced_acc: {best_ba:.4f}")

        del model, opt, sched, Xn, Xc, Xn_te, Xc_te, yt, T, tr_t, val_t, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return val_proba, test_proba

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"distill_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("distill", ds, oof, test, fold_scores)
    clear_ckpt(f"distill_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
