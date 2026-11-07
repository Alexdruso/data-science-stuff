"""Pseudo-likelihood Bayes score from a saved label-first fold LM.

For every column c: a random order of the other columns, then c last, scored under both labels;
the last segment's log-prob is log P(x_c | x_-c, y). PLL-LLR = sum_c [lp(c|1) - lp(c|0)].
The whole-sequence log-probs of those 24 passes also give the usual chain LLR at K=24.

    ~/venvs/lm/bin/python src/lm_pll.py _lm_distilgpt2_ep10 --fold 1
Writes results/<dir>_pll/f{fold}_val.npy, <dir>_k24/f{fold}_val.npy, <dir>_pllcols/f{fold}_val_cols.npy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features
from lm_leg import Segs, Vocab, prefixes, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--bs", type=int, default=256)
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
    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{a.fold}_model.pt"))
    model.eval()
    rows = np.where(folds == a.fold)[0]
    n_cols = len(RAW_COLS)
    rng = np.random.default_rng(4242)
    last = np.zeros((2, n_cols, len(rows)), dtype=np.float32)   # log P(x_c | x_-c, y)
    full = np.zeros((2, n_cols, len(rows)))                      # chain log P(x | y), one order per c
    for c in range(n_cols):
        others = np.array([x for x in range(n_cols) if x != c])
        orders = np.array([np.append(rng.permutation(others), c) for _ in rows])
        for yy in (0, 1):
            for b in range(0, len(rows), a.bs):
                sl = slice(b, b + a.bs)
                r = rows[sl]
                ids, att, score = seg.batch(r, orders[sl], [pre[yy]] * len(r))
                lmask = torch.zeros_like(score)
                lens = att.sum(1).numpy()
                for i, ri in enumerate(r):
                    lmask[i, lens[i] - seg.len[c][ri] : lens[i]] = True
                ids_c, att_c = ids.cuda(), att.cuda()
                tlp = voc.token_logp(model, ids_c, att_c)
                full[yy, c, sl] = (tlp * score[:, 1:].cuda()).sum(1).cpu().numpy()
                last[yy, c, sl] = (tlp * lmask[:, 1:].cuda()).sum(1).cpu().numpy()
                del ids_c, att_c, tlp
        print(f"  column {c + 1}/{n_cols} done", flush=True)
    torch.cuda.empty_cache()
    pll = (last[1] - last[0]).sum(0)
    lse = np.logaddexp.reduce(full, axis=1)
    k24 = lse[1] - lse[0]
    for tag, s in (("_pll", pll), ("_k24", k24)):
        out = RESULTS_DIR / f"{a.dir}{tag}"
        out.mkdir(exist_ok=True)
        np.save(out / f"f{a.fold}_val.npy", s)
        print(f"{tag:6s} solo {roc_auc_score(y[rows], s):.5f}", flush=True)
    out = RESULTS_DIR / f"{a.dir}_pllcols"
    out.mkdir(exist_ok=True)
    np.save(out / f"f{a.fold}_val_cols.npy", (last[1] - last[0]).T)
    print("corr(pll, k24)", float(np.corrcoef(pll, k24)[0, 1]), flush=True)


if __name__ == "__main__":
    main()
