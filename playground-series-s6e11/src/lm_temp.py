"""Temperature-scaled Storyteller scoring of a saved fold model (T sweep).

Original doc: how many column orders does the Storyteller score need? Rescore a saved fold model.

    ~/venvs/lm/bin/python src/lm_rescore.py _lm_distilgpt2_ep5 --fold 1 --k 16 --sub 30000

Prints AUC (and fold-subset stack gain over the EBM/tree base) for logsumexp and mean
aggregation over the first k orders, k = 1, 2, 4, ..., K.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features
from lm_fold_eval import BASE, logit
from lm_leg import Segs, Vocab, prefixes, seq_logprob, suffixes

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sub", type=int, default=30000)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    or_s = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = ref["id"].to_numpy(), ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    seg_tr = Segs(tr_s, tok)
    pre = prefixes(tok)
    used = np.unique(np.concatenate(
        [*[x.ravel() for x in seg_tr.arr + Segs(te_s, tok).arr + Segs(or_s, tok).arr],
         *[np.asarray(p) for p in pre.values()], *[np.asarray(q) for q in suffixes(tok).values()]]))
    voc = Vocab(used[used >= 0])
    model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
    model.load_state_dict(torch.load(RESULTS_DIR / a.dir / f"f{a.fold}_model.pt"))
    model.eval()
    va = np.where(folds == a.fold)[0]
    sub_pos = np.random.default_rng(5).choice(len(va), a.sub, replace=False)
    rows = va[sub_pos]
    rng = np.random.default_rng(777)
    TS = [float(t) for t in __import__("os").environ.get("TS", "0.7,0.85,1.0,1.2,1.4").split(",")]

    @torch.no_grad()
    def lp_temps(ids_, att, score):
        ids_, att, score = ids_.cuda(), att.cuda(), score.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.transformer(input_ids=ids_, attention_mask=att).last_hidden_state[:, :-1]
            logits = (h @ model.lm_head.weight[voc.used].T).float()
        tgt = voc.remap[ids_[:, 1:]].clamp(min=0)
        out = []
        for T in TS:
            lpt = torch.log_softmax(logits / T, -1).gather(-1, tgt[..., None])[..., 0]
            out.append((lpt * score[:, 1:]).sum(1).cpu())
        return torch.stack(out)

    lp = np.zeros((len(TS), 2, a.k, len(rows)))
    for kk in range(a.k):
        orders = np.array([rng.permutation(len(RAW_COLS)) for _ in rows])
        for yy in (0, 1):
            for b in range(0, len(rows), a.bs):
                sl = slice(b, b + a.bs)
                t = seg_tr.batch(rows[sl], orders[sl], [pre[yy]] * len(rows[sl]))
                lp[:, yy, kk, sl] = lp_temps(*t).numpy()
        print(f"  order {kk + 1}/{a.k} done", flush=True)
    np.save(RESULTS_DIR / a.dir / f"f{a.fold}_temp_lp_{__import__('os').environ.get('TS', 'a')}.npy", lp)
    np.save(RESULTS_DIR / a.dir / f"f{a.fold}_temp_rows.npy", rows)
    from scipy.special import logsumexp
    for i, T in enumerate(TS):
        llr = logsumexp(lp[i, 1], 0) - logsumexp(lp[i, 0], 0)
        print(f"T={T}: solo AUC {roc_auc_score(y[rows], llr):.5f}", flush=True)


if __name__ == "__main__":
    main()
