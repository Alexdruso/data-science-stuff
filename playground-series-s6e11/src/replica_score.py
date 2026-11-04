"""Score every train + test row with a generator replica (no train labels used → no CV needed).

    ~/venvs/lm/bin/python src/replica_score.py --great_format --ckpt great_e80 --k 2

Order: fold-1 val rows first (early read), then the rest of train, then test. Saves chunks to
results/_replica/{ckpt}_score/ (resumable); `--assemble` writes oof/test parquet as lm leg `replica_{ckpt}`.
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
from features import TARGET, build_features
from lm_leg import Segs, Vocab, prefixes, suffixes
from replica import score

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
CHUNK = 20000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="great_e80")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--great_format", action="store_true")
    a = ap.parse_args()
    out = RESULTS_DIR / "_replica" / f"{a.ckpt}_score"
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    te_s = te_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    seg_tr, seg_te = Segs(tr_s, tok), Segs(te_s, tok)
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg_tr.arr + seg_te.arr + Segs(or_s, tok).arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / "_replica" / f"{a.ckpt}.pt"))
    order_tr = np.concatenate([np.where(folds == 1)[0], np.where(folds != 1)[0]])
    jobs = [("tr", seg_tr, order_tr), ("te", seg_te, np.arange(seg_te.n))]
    t0 = time.time()
    for split, segs, rows in jobs:
        for c, b in enumerate(range(0, len(rows), CHUNK)):
            f = out / f"{split}_{c:03d}.npz"
            if f.exists():
                continue
            r = rows[b:b + CHUNK]
            lp = score(model, voc, segs, r, pre, a.k, a.bs, [(1.0, 0)], seed=11 + c)[(1.0, 0)]
            np.savez(f, rows=r, lp=lp.astype(np.float32))
            print(f"{split} chunk {c} done ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
