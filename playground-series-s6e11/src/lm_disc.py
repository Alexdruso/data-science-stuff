"""Discriminative fine-tune of a label-first Storyteller LM (generative -> conditional likelihood).

Start from a fold-k generative LM (results/<src>/f{k}_model.pt). Its Bayes score
    s(x) = log P(x | y=1) - log P(x | y=0) + b
is trained directly with BCE on y (same random column order for both label prefixes), plus
lam * the usual generative NLL as a regulariser. The fine-tune uses rows the generative model
never saw: the fold-k validation rows, cross-fitted in two halves (fit on half A, score half B,
and vice versa), so every scored row's own label is unused.

    ~/venvs/lm/bin/python src/lm_disc.py --src _lm_distilgpt2_ep10 --fold 1 --tag _disc
Writes results/_lm_distilgpt2<tag>/f{k}_val.npy (full fold, cross-fitted).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features
from lm_leg import Segs, Vocab, prefixes, score_rows, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def disc_fit(model: GPT2LMHeadModel, voc: Vocab, segs: Segs, rows: np.ndarray, y: np.ndarray,
             pre: dict, epochs: float, lr: float, bs: int, lam: float, neg_ratio: int,
             prior: float, rng: np.random.Generator) -> None:
    pos, neg = rows[y == 1], rows[y == 0]
    n_cols = len(RAW_COLS)
    per_ep = len(pos) * (1 + neg_ratio)
    steps = int(epochs * per_ep / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    # sampled negatives are 1/w of the population: offset keeps the logit on the true scale
    w_off = np.log(len(neg) / (len(pos) * neg_ratio))
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run = time.time(), np.zeros(2)
    print(f"  disc fit: {len(pos)} pos, {steps} steps", flush=True)
    for step in range(steps):
        npos = max(1, bs // (1 + neg_ratio))
        sel = np.concatenate([rng.choice(pos, npos), rng.choice(neg, bs - npos)])
        lab = np.r_[np.ones(npos), np.zeros(bs - npos)]
        orders = np.array([rng.permutation(n_cols) for _ in sel])
        lps = []
        for yy in (0, 1):
            ids, att, sc = segs.batch(sel, orders, [pre[yy]] * len(sel))
            ids, att, sc = ids.cuda(), att.cuda(), sc.cuda()
            lp = voc.token_logp(model, ids, att)
            lps.append(((lp * sc[:, 1:]).sum(1), (lp * att[:, 1:]).sum(1), att[:, 1:].sum(1)))
            del ids, att, sc, lp
        logit = lps[1][0] - lps[0][0] + prior - w_off
        t = torch.as_tensor(lab, device=logit.device, dtype=torch.float32)
        l_disc = bce(logit, t)
        # generative NLL under the true label (per token)
        tl = torch.where(t > 0, lps[1][1], lps[0][1])
        nt = torch.where(t > 0, lps[1][2], lps[0][2]).float()
        l_gen = -(tl / nt).mean()
        loss = l_disc + lam * l_gen
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        run += [l_disc.item(), l_gen.item()]
        del lps, logit, t, l_disc, tl, nt, l_gen, loss
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{steps} bce {run[0] / 100:.4f} gen {run[1] / 100:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run[:] = 0
    del opt, scaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_lm_distilgpt2_ep10")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=3.0, help="passes over the half's positives")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--k_val", type=int, default=3)
    ap.add_argument("--tag", default="_disc")
    ap.add_argument("--fit_on", default="valhalves", choices=["valhalves", "train"],
                    help="train: fold-train positives + the negatives the generative fit never saw")
    ap.add_argument("--sub", type=int, default=0, help="smoke test on N val rows")
    a = ap.parse_args()
    out = RESULTS_DIR / f"_lm_{a.model}{a.tag}"
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    assert (ref["id"].to_numpy() == tr_s["id"].to_numpy()).all()
    y = ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    seg = Segs(tr_s, tok)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    pre = prefixes(tok)
    used = np.unique(np.concatenate([*[x.ravel() for x in seg.arr + Segs(te_s, tok).arr
                                       + Segs(or_s, tok).arr],
                                     *[np.asarray(p) for p in pre.values()],
                                     *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    k = a.fold
    tr_k = folds != k
    prior = float(np.log(y[tr_k].mean() / (1 - y[tr_k].mean())))
    va = np.where(folds == k)[0]
    rng = np.random.default_rng(77)
    if a.sub:
        va = np.sort(rng.choice(va, a.sub, replace=False))
    half = np.zeros(len(va), int)
    for c in (0, 1):  # stratified halves
        idx = np.where(y[va] == c)[0]
        half[rng.permutation(idx)[: len(idx) // 2]] = 1
    sv = np.zeros(len(va))
    base = np.zeros(len(va))
    if a.fit_on == "train":
        tr_idx = np.where(tr_k)[0]
        keep = (y[tr_idx] == 1) | (np.random.default_rng(1000 + k).random(len(tr_idx)) < 0.25)
        unseen_neg = tr_idx[(~keep) & (y[tr_idx] == 0)]
        fit_rows = np.concatenate([tr_idx[y[tr_idx] == 1], unseen_neg])
        model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
        model.load_state_dict(torch.load(RESULTS_DIR / a.src / f"f{k}_model.pt"))
        disc_fit(model, voc, seg, fit_rows, y[fit_rows], pre, a.epochs, a.lr, a.bs, a.lam,
                 a.neg_ratio, prior, np.random.default_rng(10))
        sv = score_rows(model, voc, seg, va, pre, a.k_val, 4 * a.bs * 2, k) + prior
        base = np.load(RESULTS_DIR / a.src / f"f{k}_val.npy") if not a.sub else sv * 0
        print(f"fold {k}: disc {roc_auc_score(y[va], sv):.5f}", flush=True)
        del model
        torch.cuda.empty_cache()
        if not a.sub:
            np.save(out / f"f{k}_val.npy", sv)
        return
    for h in (0, 1):
        fit_rows, sc_rows = va[half == h], va[half != h]
        model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
        model.load_state_dict(torch.load(RESULTS_DIR / a.src / f"f{k}_model.pt"))
        if h == 0:  # the untouched generative score on the same orders, for a paired read
            base = score_rows(model, voc, seg, va, pre, a.k_val, 4 * a.bs * 2, k) + prior
            print(f"  base gen LM val auc {roc_auc_score(y[va], base):.5f}", flush=True)
        disc_fit(model, voc, seg, fit_rows, y[fit_rows], pre, a.epochs, a.lr, a.bs, a.lam,
                 a.neg_ratio, prior, np.random.default_rng(10 + h))
        s = score_rows(model, voc, seg, sc_rows, pre, a.k_val, 4 * a.bs * 2, k) + prior
        sv[half != h] = s
        print(f"  half {h}: disc auc {roc_auc_score(y[sc_rows], s):.5f} vs gen "
              f"{roc_auc_score(y[sc_rows], base[half != h]):.5f}", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"fold {k}: disc {roc_auc_score(y[va], sv):.5f} gen {roc_auc_score(y[va], base):.5f}",
          flush=True)
    if a.sub:
        return
    np.save(out / f"f{k}_val.npy", sv)
    np.save(out / f"f{k}_val_gen.npy", base)


if __name__ == "__main__":
    main()
