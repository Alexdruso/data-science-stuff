"""Per-column evidence from a saved Storyteller fold model.

For a column order o, log P(x | y) = sum_j log P(seg_j | y, segs before j). This saves, per row,
the per-column log-likelihood *ratio* LLR_j = log P(seg_j | 1, ctx) - log P(seg_j | 0, ctx),
averaged over K random orders (shape n x 24), plus the per-order totals (2 x K x n).
The stacker can then re-weight columns the LM is over/under-confident on.

    ~/venvs/lm/bin/python src/lm_colscore.py _lm_distilgpt2_ep10 --fold 1 --k 4 [--test]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features
from lm_leg import Segs, Vocab, col_lp, prefixes, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te_s = te_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    seg_tr, seg_te = Segs(tr_s, tok), Segs(te_s, tok)
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg_tr.arr + seg_te.arr + Segs(or_s, tok).arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{a.fold}_model.pt"))
    model.eval()
    jobs = [("val", seg_tr, np.where(folds == a.fold)[0])]
    if a.test:
        jobs.append(("test", seg_te, np.arange(seg_te.n)))
    n_cols = len(RAW_COLS)
    for split, segs, rows in jobs:
        rng = np.random.default_rng(4242 + a.fold)
        llr = np.zeros((len(rows), n_cols), dtype=np.float32)
        tot = np.zeros((2, a.k, len(rows)), dtype=np.float32)
        t0 = time.time()
        for kk in range(a.k):
            orders = np.array([rng.permutation(n_cols) for _ in rows])
            for b in range(0, len(rows), a.bs):
                sl = slice(b, b + a.bs)
                c1 = col_lp(model, voc, segs, rows[sl], orders[sl], pre[1])
                c0 = col_lp(model, voc, segs, rows[sl], orders[sl], pre[0])
                llr[sl] += (c1 - c0) / a.k
                tot[1, kk, sl], tot[0, kk, sl] = c1.sum(1), c0.sum(1)
            print(f"  {split} order {kk + 1}/{a.k} ({time.time() - t0:.0f}s)", flush=True)
        np.save(RESULTS_DIR / a.dir / f"f{a.fold}_{split}_colllr.npy", llr)
        np.save(RESULTS_DIR / a.dir / f"f{a.fold}_{split}_coltot.npy", tot)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
