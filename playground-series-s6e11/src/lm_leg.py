"""GReaT-family "Storyteller" leg: label-first GPT-2 over the rows as text, scored by Bayes.

Playground data is sampled by GReaT (GPT-2 on "col is value, ..." text, random column order,
label first). This leg fits the same model family per fold and scores each row as
    log P(x | y=1) - log P(x | y=0)  (logsumexp over K column orders)  + prior log-odds.

Values keep the CSV text formatting (read as strings). Column names are shortened to cut the
token count ~2x (the 6 GB card is the budget); negatives are subsampled for training only
(P(x|y) is unaffected by the class ratio; the true prior is added back at scoring).

Runs in the overlay venv that has `transformers`:
    ~/venvs/lm/bin/python src/lm_leg.py --folds 1 [--model distilgpt2] [--epochs 3]
Per-fold outputs are cached in results/_<name>/f{k}_{val,test}.npy; assembled when all 5 exist.
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

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

SHORT = {
    "amount_usd": "amount", "hours_since_last_txn": "hours since last", "txn_count_last_24h":
    "txn count", "distance_from_home_km": "distance", "card_age_months": "card age",
    "customer_age": "customer age", "account_balance_usd": "balance", "cvv_retry_count": "cvv retry",
    "velocity_score": "velocity", "time_of_day_hour": "hour", "day_of_week": "weekday",
    "merchant_risk_score": "merchant risk", "prior_disputes": "disputes", "merchant_category":
    "category", "card_type": "card", "auth_method": "auth", "channel": "channel", "device_type":
    "device", "is_foreign_transaction": "foreign", "is_new_merchant": "new merchant", "used_vpn":
    "vpn", "ip_country_mismatch": "ip mismatch", "billing_shipping_mismatch": "billing mismatch",
    "is_ai_generated_scam_attempt": "ai scam",
}


class Segs:
    """Token ids of every " name value," segment, per row and column (padded with -1)."""

    def __init__(self, df: pl.DataFrame, tok: AutoTokenizer) -> None:
        self.arr: list[np.ndarray] = []
        self.len: list[np.ndarray] = []
        for c in RAW_COLS:
            vals = df[c].to_numpy()
            uniq, inv = np.unique(vals, return_inverse=True)
            toks = [tok(f" {SHORT[c]} {v},").input_ids for v in uniq]
            width = max(len(t) for t in toks)
            table = np.full((len(uniq), width), -1, dtype=np.int64)
            for i, t in enumerate(toks):
                table[i, : len(t)] = t
            self.arr.append(table[inv])
            self.len.append(np.array([len(t) for t in toks])[inv])
        self.n = len(df)

    def batch(self, rows: np.ndarray, orders: np.ndarray, prefix: list[list[int]]
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequences [prefix | segments in order]; returns ids, attention mask, score mask."""
        seqs, starts = [], []
        for r, o, p in zip(rows, orders, prefix):
            body = [self.arr[c][r, : self.len[c][r]] for c in o]
            seqs.append(np.concatenate([np.asarray(p, dtype=np.int64), *body]))
            starts.append(len(p))
        width = max(len(s) for s in seqs)
        ids = np.zeros((len(seqs), width), dtype=np.int64)
        att = np.zeros((len(seqs), width), dtype=np.int64)
        score = np.zeros((len(seqs), width), dtype=bool)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            ids[i, : len(s)] = s
            att[i, : len(s)] = 1
            score[i, st : len(s)] = True
        return torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(score)


class Vocab:
    """Softmax restricted to the tokens the rows can contain (tied to the GPT-2 embedding)."""

    def __init__(self, used: np.ndarray) -> None:
        self.used = torch.as_tensor(np.sort(used)).cuda()
        remap = np.full(int(used.max()) + 1, -1, dtype=np.int64)
        remap[np.sort(used)] = np.arange(len(used))
        self.remap = torch.as_tensor(remap).cuda()

    def token_logp(self, model: GPT2LMHeadModel, ids: torch.Tensor, att: torch.Tensor
                   ) -> torch.Tensor:
        """log P(ids[:, t+1] | ids[:, :t+1]) under the restricted softmax, shape (B, L-1)."""
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            logits = h @ model.lm_head.weight[self.used].T
        lp = torch.log_softmax(logits.float(), -1)
        tgt = self.remap[ids[:, 1:]].clamp(min=0)
        return lp.gather(-1, tgt[..., None])[..., 0]


def seq_logprob(model: GPT2LMHeadModel, voc: Vocab, ids: torch.Tensor, att: torch.Tensor,
                score: torch.Tensor) -> torch.Tensor:
    """Sum of log P(token) over the score-masked positions."""
    ids, att, score = ids.cuda(), att.cuda(), score.cuda()
    lp = voc.token_logp(model, ids, att)
    out = (lp * score[:, 1:]).sum(1).cpu()
    del ids, att, score, lp
    return out


def prefixes(tok: AutoTokenizer) -> dict[int, list[int]]:
    bos = tok.bos_token_id
    return {y: [bos, *tok(f"fraud {y},").input_ids] for y in (0, 1)}


@torch.no_grad()
def score_rows(model: GPT2LMHeadModel, voc: Vocab, segs: Segs, rows: np.ndarray, pre: dict[int, list[int]],
               k_orders: int, bs: int, seed: int) -> np.ndarray:
    """log P(x|1) - log P(x|0), logsumexp over k fixed random orders per row."""
    model.eval()
    rng = np.random.default_rng(seed)
    n_cols = len(RAW_COLS)
    lp = np.zeros((2, k_orders, len(rows)))
    for kk in range(k_orders):
        orders = np.array([rng.permutation(n_cols) for _ in range(len(rows))])
        for y in (0, 1):
            for b in range(0, len(rows), bs):
                sl = slice(b, b + bs)
                ids, att, sc = segs.batch(rows[sl], orders[sl], [pre[y]] * len(rows[sl]))
                lp[y, kk, sl] = seq_logprob(model, voc, ids, att, sc).numpy()
    lse = np.logaddexp.reduce(lp, axis=1)
    return lse[1] - lse[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--neg_frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k_val", type=int, default=4)
    ap.add_argument("--k_test", type=int, default=2)
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--val_sub", type=int, default=0, help="score only N val rows (pilot)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = f"lm_{a.model}{a.tag}"
    out = RESULTS_DIR / f"_{name}"
    out.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tr_s = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False)
    te_s = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    tr_s = tr_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te_s = te_s.with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    assert (ref["id"].to_numpy() == tr_s["id"].to_numpy()).all()
    y = ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    t0 = time.time()
    seg_tr, seg_te = Segs(tr_s, tok), Segs(te_s, tok)
    print(f"tokenized in {time.time() - t0:.0f}s", flush=True)
    pre = prefixes(tok)
    n_cols = len(RAW_COLS)
    used = np.unique(np.concatenate([*[a.ravel() for a in seg_tr.arr + seg_te.arr],
                                     *[np.asarray(p) for p in pre.values()]]))
    voc = Vocab(used[used >= 0])
    print(f"restricted vocab: {len(voc.used)} tokens", flush=True)

    for k in a.folds:
        if (out / f"f{k}_val.npy").exists():
            continue
        rng = np.random.default_rng(1000 + k)
        tr_idx = np.where(folds != k)[0]
        va_idx = np.where(folds == k)[0]
        keep = (y[tr_idx] == 1) | (rng.random(len(tr_idx)) < a.neg_frac)
        fit_idx = tr_idx[keep]
        prior = np.log(y[tr_idx].mean() / (1 - y[tr_idx].mean()))
        model = GPT2LMHeadModel.from_pretrained(a.model).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = int(a.epochs * len(fit_idx) / a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
        scaler = torch.amp.GradScaler()
        print(f"fold {k}: fit {len(fit_idx)} rows ({y[fit_idx].sum()} pos), {steps} steps",
              flush=True)
        model.train()
        t0 = time.time()
        step, run_loss = 0, 0.0
        while step < steps:
            perm = rng.permutation(fit_idx)
            for b in range(0, len(perm) - a.bs + 1, a.bs):
                if step >= steps:
                    break
                rows = perm[b : b + a.bs]
                orders = np.array([rng.permutation(n_cols) for _ in rows])
                ids, att, _ = seg_tr.batch(rows, orders, [pre[int(y[r])] for r in rows])
                ids, att = ids.cuda(), att.cuda()
                lp = voc.token_logp(model, ids, att)
                m = att[:, 1:].float()
                loss = -(lp * m).sum() / m.sum()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                run_loss += loss.item()
                step += 1
                del ids, att, lp, m, loss
                if step % 500 == 0:
                    print(f"  step {step}/{steps} loss {run_loss / 500:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    run_loss = 0.0
        sub = va_idx if not a.val_sub else np.random.default_rng(5).choice(va_idx, a.val_sub,
                                                                             replace=False)
        t0 = time.time()
        sv = score_rows(model, voc, seg_tr, sub, pre, a.k_val, 4 * a.bs, seed=k) + prior
        print(f"  fold {k}: val auc {roc_auc_score(y[sub], sv):.5f} on {len(sub)} rows "
              f"({time.time() - t0:.0f}s)", flush=True)
        if a.val_sub:
            np.save(out / f"f{k}_valsub_idx.npy", sub)
            np.save(out / f"f{k}_valsub.npy", sv)
        else:
            np.save(out / f"f{k}_val.npy", sv)
            if not a.no_test:
                st = score_rows(model, voc, seg_te, np.arange(seg_te.n), pre, a.k_test, 4 * a.bs,
                                seed=100 + k) + prior
                np.save(out / f"f{k}_test.npy", st)
            torch.save(model.state_dict(), out / f"f{k}_model.pt")
        del model, opt
        torch.cuda.empty_cache()

    if all((out / f"f{k}_val.npy").exists() and (out / f"f{k}_test.npy").exists()
           for k in range(1, 6)):
        from cv_results import save_cv_result

        from data_science_stuff.kaggle.io import write_submission
        from data_science_stuff.kaggle.predictions import save_predictions

        oof = np.zeros(len(y))
        te = np.zeros(seg_te.n)
        for k in range(1, 6):
            oof[folds == k] = np.load(out / f"f{k}_val.npy")
            te += np.load(out / f"f{k}_test.npy") / 5
        auc = float(roc_auc_score(y, oof))
        print(f"{name}: OOF AUC {auc:.5f}")
        sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
        save_predictions(RESULTS_DIR, name, oof=sig(oof), test=sig(te),
                         train_ids=tr_s["id"].to_numpy(), test_ids=te_s["id"].to_numpy(),
                         folds=folds)
        save_cv_result(RESULTS_DIR, name, [], auc, metric_name="auc", params=vars(a))
        write_submission(SUBMISSIONS_DIR, f"{name}.csv", te_s["id"].to_numpy(), TARGET, sig(te))


if __name__ == "__main__":
    main()
