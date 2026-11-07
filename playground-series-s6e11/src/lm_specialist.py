"""Class-specialist fine-tune of a saved joint label-first LM.

Copy A: fine-tuned on the fold-train positives only (prefix "fraud 1,"); copy B: on an equal-step
sample of fold-train negatives the generative fit never saw (prefix "fraud 0,"). Score
LLR = log P_A(x | 1) - log P_B(x | 0) on the fold's val rows; paired reference = the joint model
on the same rows and orders.

    ~/venvs/lm/bin/python src/lm_specialist.py _lm_distilgpt2_ep10 --fold 1 --epochs 2 --lr 2e-5
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features
from lm_leg import Segs, Vocab, fit_lm, prefixes, seq_logprob, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


@torch.no_grad()
def lp_rows(model: GPT2LMHeadModel, voc: Vocab, seg: Segs, rows: np.ndarray,
            orders: np.ndarray, prefix: list[int], bs: int) -> np.ndarray:
    model.eval()
    out = np.zeros(len(rows))
    for b in range(0, len(rows), bs):
        sl = slice(b, b + bs)
        t = seg.batch(rows[sl], orders[sl], [prefix] * len(rows[sl]))
        out[sl] = seq_logprob(model, voc, *t).numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--neg_mult", type=float, default=1.0, help="negatives per positive (B fit)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = ref[TARGET].to_numpy().astype(int)
    assert (tr_s["id"].to_numpy() == ref["id"].to_numpy()).all()
    folds = fold_assignment(y)
    seg = Segs(tr_s, tok)
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg.arr + Segs(te_s, tok).arr + Segs(or_s, tok).arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    k = a.fold
    # reproduce the generative fit's negative subsample (lm_leg: rng 1000+k, neg_frac .25)
    rng = np.random.default_rng(1000 + k)
    tr_idx = np.where(folds != k)[0]
    keep = (y[tr_idx] == 1) | (rng.random(len(tr_idx)) < 0.25)
    pos = tr_idx[y[tr_idx] == 1]
    unseen_neg = tr_idx[(~keep) & (y[tr_idx] == 0)]
    r2 = np.random.default_rng(77)
    neg = r2.choice(unseen_neg, int(a.neg_mult * len(pos)), replace=False)
    va = np.where(folds == k)[0]
    ro = np.random.default_rng(555)
    orders = [np.array([ro.permutation(len(RAW_COLS)) for _ in va]) for _ in range(a.k)]
    sbs = 256

    base = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    base.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{k}_model.pt"))
    ref_lp = np.stack([[lp_rows(base, voc, seg, va, o, pre[yy], sbs) for o in orders]
                       for yy in (0, 1)])  # (2, K, n)
    print("reference scored", flush=True)
    spec = np.zeros((2, a.k, len(va)))
    for yy, rows in ((1, pos), (0, neg)):
        m = copy.deepcopy(base)
        fit_lm(m, voc, seg, rows, np.full(len(rows), yy), pre, a.epochs, a.lr, a.bs,
               np.random.default_rng(10 + yy))
        spec[yy] = np.stack([lp_rows(m, voc, seg, va, o, pre[yy], sbs) for o in orders])
        del m
        torch.cuda.empty_cache()
        print(f"specialist {yy} scored", flush=True)
    del base
    torch.cuda.empty_cache()

    def llr(lp: np.ndarray) -> np.ndarray:
        lse = np.logaddexp.reduce(lp, axis=1)
        return lse[1] - lse[0]
    s_ref, s_spec = llr(ref_lp), llr(spec)
    mixA = llr(np.stack([ref_lp[0], spec[1]]))  # only the positive specialist
    for tag, s in (("_specref", s_ref), (f"_spec{a.tag}", s_spec), (f"_specpos{a.tag}", mixA)):
        out = RESULTS_DIR / f"{a.dir}{tag}"
        out.mkdir(exist_ok=True)
        np.save(out / f"f{k}_val.npy", s)
        print(f"{tag:14s} solo {roc_auc_score(y[va], s):.5f}", flush=True)


if __name__ == "__main__":
    main()
