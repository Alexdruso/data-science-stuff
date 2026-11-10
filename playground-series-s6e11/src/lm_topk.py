"""Top-k truncation evidence from a saved fold LM (PR-16).

GReaT samples with top-k 50: a value token outside the generator's top 50 under a label's
context cannot be produced with that label. The LLR only sees this smoothly (log p small).
For the same rows/orders as lm_rescore_names (rng 777), record per value token the rank of the
true token in the restricted vocabulary under each label, then per row and label:
  n_out[k]   = #value tokens with rank >= k (k in 10, 20, 50, 100)
  max_lrank  = max log(1 + rank)
aggregated over orders by mean and min (the row is producible if *some* order keeps every
token in the top k).

    ~/venvs/lm/bin/python src/lm_topk.py _lm_distilgpt2_ep10 --fold 1 --k 3

Writes results/<dir>_topk/f{fold}_val_feats.npz (feature matrix + names + llr).
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
KS = (10, 20, 50, 100)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--bs", type=int, default=128)
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
    name_ids = [tok(" " + SHORT[c]).input_ids for c in RAW_COLS]
    n_name = np.array([len(n) for n in name_ids])

    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{a.fold}_model.pt"))
    model.eval()
    W = model.lm_head.weight[voc.used]
    rows = np.where(folds == a.fold)[0]
    n = len(rows)
    rng = np.random.default_rng(777)
    lp = np.zeros((2, a.k, n))
    nout = np.zeros((2, a.k, len(KS), n))
    mxr = np.zeros((2, a.k, n))
    # per-row "impossible under y but possible under 1-y" counts at k=50 (same order, same position)
    excl = np.zeros((2, a.k, n))
    for kk in range(a.k):
        orders = np.array([rng.permutation(len(RAW_COLS)) for _ in rows])
        for b in range(0, n, a.bs):
            sl = slice(b, b + a.bs)
            r, o = rows[sl], orders[sl]
            rk = []
            vms = []
            for yy in (0, 1):
                ids_t, att, score = segs_batch = seg_tr.batch(r, o, [pre[yy]] * len(r))
                vmask = score.clone()
                p0 = len(pre[yy])
                for i, (ri, oi) in enumerate(zip(r, o)):
                    pos = p0
                    for c in oi:
                        vmask[i, pos : pos + n_name[c]] = False
                        pos += seg_tr.len[c][ri]
                ids_c, att_c = ids_t.cuda(), att.cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    h = model.transformer(input_ids=ids_c, attention_mask=att_c).last_hidden_state[:, :-1]
                    logits = (h @ W.T).float()
                tgt = voc.remap[ids_c[:, 1:]].clamp(min=0)
                tl = logits.gather(-1, tgt[..., None])
                rank = (logits > tl).sum(-1).float()  # 0 = top-1
                tlp = (tl[..., 0] - torch.logsumexp(logits, -1))
                sc = score[:, 1:].cuda()
                vm = vmask[:, 1:].cuda()
                lp[yy, kk, sl] = (tlp * sc).sum(1).cpu().numpy()
                rmask = torch.where(vm, rank, torch.zeros_like(rank))
                for j, kv in enumerate(KS):
                    nout[yy, kk, j, sl] = ((rmask >= kv) & vm).sum(1).cpu().numpy()
                mxr[yy, kk, sl] = torch.log1p(rmask).max(1).values.cpu().numpy()
                rk.append(rmask)
                vms.append(vm)
                del ids_c, att_c, h, logits, tgt, tl, rank, tlp, sc
            # both label prefixes have equal length here ("fraud 0," / "fraud 1,") -> aligned positions
            if rk[0].shape == rk[1].shape:
                excl[0, kk, sl] = (((rk[0] >= 50) & (rk[1] < 50)) & vms[0]).sum(1).cpu().numpy()
                excl[1, kk, sl] = (((rk[1] >= 50) & (rk[0] < 50)) & vms[1]).sum(1).cpu().numpy()
            del rk, vms
        print(f"  order {kk + 1}/{a.k} done", flush=True)
    torch.cuda.empty_cache()
    lse = np.logaddexp.reduce(lp, axis=1)
    llr = lse[1] - lse[0]
    ys = y[rows]
    print(f"LLR solo {roc_auc_score(ys, llr):.5f}", flush=True)
    feats, names = [], []
    for agg, f in (("mean", np.mean), ("min", np.min)):
        for j, kv in enumerate(KS):
            for yy in (0, 1):
                feats.append(f(nout[yy, :, j], axis=0)); names.append(f"nout{kv}_y{yy}_{agg}")
            feats.append(f(nout[0, :, j], axis=0) - f(nout[1, :, j], axis=0)); names.append(f"nout{kv}_d_{agg}")
        for yy in (0, 1):
            feats.append(f(mxr[yy], axis=0)); names.append(f"maxlrank_y{yy}_{agg}")
            feats.append(f(excl[yy], axis=0)); names.append(f"excl_y{yy}_{agg}")
    F = np.column_stack(feats)
    for nm, col in zip(names, F.T):
        if col.std() > 0:
            print(f"  {nm:22s} mean {col.mean():.3f}  auc {roc_auc_score(ys, col):.4f}", flush=True)
    out = RESULTS_DIR / f"{a.dir}_topk"
    out.mkdir(exist_ok=True)
    np.savez(out / f"f{a.fold}_val_feats.npz", F=F, names=np.array(names), llr=llr, rows=rows)


if __name__ == "__main__":
    main()
