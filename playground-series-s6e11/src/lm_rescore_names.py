"""Rescore a saved fold LM with and without the column-name tokens in the Bayes score.

The column order is our own random shuffle, so log P(next column name | label, prefix) holds no
label information in truth; its label-conditional part is fit noise added to the LLR. This
scores the same rows / orders / model twice: all body tokens (the stored leg) and value tokens
only (name tokens masked out; the trailing comma = "value ends here" is kept).

    ~/venvs/lm/bin/python src/lm_rescore_names.py _lm_distilgpt2_ep10 --fold 1 --k 3

Writes results/<dir>_names/f{fold}_val.npy (all tokens) and <dir>_nonames/f{fold}_val.npy.
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
from lm_leg import SHORT, Segs, Vocab, prefixes, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te_s = te_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = ref["id"].to_numpy(), ref[TARGET].to_numpy().astype(int)
    assert (tr_s["id"].to_numpy() == ids).all()
    folds = fold_assignment(y)
    seg_tr, seg_te = Segs(tr_s, tok), Segs(te_s, tok)
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg_tr.arr + seg_te.arr + Segs(or_s, tok).arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    # name-token count per column; check every segment starts with exactly those ids
    name_ids = [tok(" " + SHORT[c]).input_ids for c in RAW_COLS]
    for c, nid in enumerate(name_ids):
        assert (seg_tr.arr[c][:, : len(nid)] == np.asarray(nid)).all(), RAW_COLS[c]
    n_name = np.array([len(n) for n in name_ids])
    print("name tokens per column", dict(zip(RAW_COLS, n_name.tolist())), flush=True)

    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{a.fold}_model.pt"))
    model.eval()
    segs = seg_tr if a.split == "val" else seg_te
    rows = np.where(folds == a.fold)[0] if a.split == "val" else np.arange(seg_te.n)
    rng = np.random.default_rng(777)
    lp = np.zeros((2, 2, a.k, len(rows)))  # [mask: all/values][label][order][row]
    for kk in range(a.k):
        orders = np.array([rng.permutation(len(RAW_COLS)) for _ in rows])
        for yy in (0, 1):
            for b in range(0, len(rows), a.bs):
                sl = slice(b, b + a.bs)
                r, o = rows[sl], orders[sl]
                ids_t, att, score = segs.batch(r, o, [pre[yy]] * len(r))
                vmask = score.clone()
                p0 = len(pre[yy])
                for i, (ri, oi) in enumerate(zip(r, o)):
                    pos = p0
                    for c in oi:
                        vmask[i, pos : pos + n_name[c]] = False
                        pos += segs.len[c][ri]
                ids_c, att_c = ids_t.cuda(), att.cuda()
                tlp = voc.token_logp(model, ids_c, att_c)
                lp[0, yy, kk, sl] = (tlp * score[:, 1:].cuda()).sum(1).cpu().numpy()
                lp[1, yy, kk, sl] = (tlp * vmask[:, 1:].cuda()).sum(1).cpu().numpy()
                del ids_c, att_c, tlp
        print(f"  order {kk + 1}/{a.k} done", flush=True)
    torch.cuda.empty_cache()
    for m, tag in ((0, "_names"), (1, "_nonames")):
        lse = np.logaddexp.reduce(lp[m], axis=1)
        s = lse[1] - lse[0]
        out = RESULTS_DIR / f"{a.dir}{tag}"
        out.mkdir(exist_ok=True)
        np.save(out / f"f{a.fold}_{a.split}.npy", s)
        if a.split == "val":
            print(f"{tag:10s} solo {roc_auc_score(y[rows], s):.5f}", flush=True)
    name_part = (np.logaddexp.reduce(lp[0], 1) - np.logaddexp.reduce(lp[1], 1))
    print("name-term LLR sd", float((name_part[1] - name_part[0]).std()), flush=True)


if __name__ == "__main__":
    main()
