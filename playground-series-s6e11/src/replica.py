"""Generator replica: GPT-2 fit on the ORIGINAL rows only (GReaT text), scored as the generator samples.

Playground rows were sampled from a GPT-2 fine-tuned on the original 20k rows, label first, at
temperature ~0.7 with top-k. A replica trained on the original rows uses no train labels, so its
scores on train rows are leak-free and need no CV. Scoring = Bayes over the label prompt with the
row's log-probability under the tempered (and optionally top-k truncated) next-token distribution,
logsumexp over K column orders.

    ~/venvs/lm/bin/python src/replica.py --great_format --epochs 10 20 40 --n 30000
Saves results/_replica/{tag}_e{E}.pt and prints, per checkpoint and scoring mode, the AUC on a
fold-1 val subset and the mean log-likelihood of the rows (generator-match criterion).
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
from lm_leg import Segs, Vocab, fit_lm, prefixes, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


@torch.no_grad()
def seq_lp_modes(model: GPT2LMHeadModel, voc: Vocab, ids: torch.Tensor, att: torch.Tensor,
                 score: torch.Tensor, modes: list) -> dict:
    """Row log-prob under each (temperature, top-k) next-token distribution; one forward pass."""
    ids, att, score = ids.cuda(), att.cuda(), score.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        raw = (h @ model.lm_head.weight[voc.used].T).float()
    tgt = voc.remap[ids[:, 1:]].clamp(min=0)[..., None]
    res = {}
    for temp, topk in modes:
        logits = raw / temp
        if topk:
            kth = logits.topk(topk, -1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, -1e4)
        tok = torch.log_softmax(logits, -1).gather(-1, tgt)[..., 0].clamp(min=-1e4)
        res[(temp, topk)] = (tok * score[:, 1:]).sum(1).cpu().numpy()
        del logits, tok
    del ids, att, score, h, raw, tgt
    return res


def score(model, voc, segs, rows, pre, k_orders, bs, modes, seed=7):
    rng = np.random.default_rng(seed)
    n_cols = len(RAW_COLS)
    lp = {m: np.zeros((2, k_orders, len(rows))) for m in modes}
    model.eval()
    for kk in range(k_orders):
        orders = np.array([rng.permutation(n_cols) for _ in rows])
        for yy in (0, 1):
            for b in range(0, len(rows), bs):
                sl = slice(b, b + bs)
                t = segs.batch(rows[sl], orders[sl], [pre[yy]] * len(rows[sl]))
                r = seq_lp_modes(model, voc, *t, modes)
                for m in modes:
                    lp[m][yy, kk, sl] = r[m]
    return lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=float, nargs="+", default=[10, 20, 40])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tag", default="great")
    ap.add_argument("--great_format", action="store_true")
    a = ap.parse_args()
    out = RESULTS_DIR / "_replica"
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    y = ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    seg_tr, seg_or = Segs(tr_s, tok), Segs(or_s, tok)
    y_or = or_s[TARGET].cast(pl.Float64).cast(pl.Int64).to_numpy()
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg_tr.arr + Segs(te_s, tok).arr + seg_or.arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    va = np.where(folds == 1)[0]
    rows = np.sort(np.random.default_rng(5).choice(va, a.n, replace=False))
    np.save(out / "rows_f1.npy", rows)
    ys = y[rows]
    prior = np.log(y_or.mean() / (1 - y_or.mean()))
    modes = [(1.0, 0), (0.7, 0), (0.7, 50)]
    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    done = 0.0
    rng = np.random.default_rng(0)
    for e in a.epochs:
        t0 = time.time()
        ck = out / f"{a.tag}_e{e:g}.pt"
        if ck.exists():
            model.load_state_dict(torch.load(ck))
        else:
            fit_lm(model, voc, seg_or, np.arange(seg_or.n), y_or, pre, e - done, a.lr, a.bs, rng)
            torch.save(model.state_dict(), ck)
        done = e
        torch.cuda.empty_cache()
        if (out / f"{a.tag}_e{e:g}_T0.7_k50_f1sub.npy").exists():
            continue
        lp = score(model, voc, seg_tr, rows, pre, a.k, 2 * a.bs, modes)
        for m in modes:
            lse = np.logaddexp.reduce(lp[m], axis=1)
            s = lse[1] - lse[0]
            ll = np.where(ys == 1, lse[1] + np.log(y_or.mean()), lse[0] + np.log(1 - y_or.mean()))
            np.save(out / f"{a.tag}_e{e:g}_T{m[0]}_k{m[1]}_f1sub.npy", s + prior)
            print(f"epoch {e:g} T={m[0]} topk={m[1]}: AUC {roc_auc_score(ys, s):.5f}  "
                  f"mean loglik {ll.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
