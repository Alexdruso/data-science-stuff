"""Faithful GReaT replica: GPT-2 on the ORIGINAL rows, trained the way be_great trains.

The earlier replica (`replica.py`) was fit label-first with our own schedule. be_great does not:
every row is the text "col is value, col is value, ..." with ALL fields (the label among them) in
a fresh random order, HF-Trainer defaults (AdamW, lr 5e-5 linear decay, no warmup, wd 0), loss on
every token including the EOS padding. Only sampling is label-first ("is_fraud is 1, ...").

No train label is used, so the AUC of the replica's Bayes score on train rows needs no CV: it is a
direct measure of how close the replica is to the generator (the true generator would be optimal).

    ~/venvs/lm/bin/python src/replica_any.py --epochs 60 --eval 20 40 60 [--model distilgpt2] [--bs 16]
Saves results/_replica/any_<model>_e{E}.pt and any_<model>_e{E}_f1sub.npy (LLR on the same 30k
fold-1 rows as replica.py's e10-e40 files; the label is written "0"/"1" as in the original CSV).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
OUT = RESULTS_DIR / "_replica"


def field_texts(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """[n, len(cols)] array of "col is value" strings (values exactly as written in the CSV)."""
    return np.column_stack([np.char.add(f"{c} is ", np.char.strip(df[c].to_numpy().astype(str)))
                            for c in cols])


def row_texts(fields: np.ndarray, rng: np.random.Generator, first: str | None = None) -> list[str]:
    """One random field order per row; `first` (a label field) is put in front when given."""
    out = []
    for f in fields:
        body = ", ".join(f[rng.permutation(len(f))])
        out.append(body if first is None else f"{first}, {body}")
    return out


class Used:
    """Softmax over the tokens the rows can contain (+ EOS); a 50k-token softmax OOMs the 6 GB card."""

    def __init__(self, tok: AutoTokenizer, frames: list[pl.DataFrame], cols: list[str]) -> None:
        pieces = ["is_fraud is 0", "is_fraud is 1"]
        for c in cols:
            vals = np.unique(np.concatenate([np.char.strip(f[c].to_numpy().astype(str)) for f in frames]))
            pieces += [f"{c} is {v}" for v in vals]
        ids: set[int] = {tok.eos_token_id}
        for i in range(0, len(pieces), 20000):
            chunk = pieces[i : i + 20000]
            for x in tok(chunk).input_ids + tok([", " + q for q in chunk]).input_ids:
                ids.update(x)
        self.ids = torch.tensor(sorted(ids)).cuda()
        remap = torch.full((len(tok),), -1, dtype=torch.long)
        remap[self.ids.cpu()] = torch.arange(len(ids))
        self.remap = remap.cuda()

    def logits(self, model: torch.nn.Module, ids: torch.Tensor, att: torch.Tensor | None = None) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        return h.float() @ model.lm_head.weight[self.ids].T

    def targets(self, ids: torch.Tensor) -> torch.Tensor:
        t = self.remap[ids[:, 1:]]
        assert (t >= 0).all()
        return t


@torch.no_grad()
def bayes_llr(model: torch.nn.Module, tok: AutoTokenizer, used: Used, fields: np.ndarray, k: int, bs: int,
              temps: tuple[float, ...] = (1.0, 0.7), seed: int = 7) -> dict[float, np.ndarray]:
    """log P(row | is_fraud 1) - log P(row | is_fraud 0) per temperature, logsumexp over k orders."""
    model.eval()
    rng = np.random.default_rng(seed)
    n1 = len(tok("is_fraud is 1,").input_ids)
    assert n1 == len(tok("is_fraud is 0,").input_ids)
    lp = {t: np.zeros((2, k, len(fields))) for t in temps}
    for kk in range(k):
        bodies = row_texts(fields, rng)
        order = np.argsort([len(b) for b in bodies])
        for b in range(0, len(order), bs):
            sel = order[b : b + bs]
            for yy in (0, 1):
                enc = tok([f"is_fraud is {yy}, {bodies[i]}" for i in sel], return_tensors="pt", padding=True)
                ids, att = enc.input_ids.cuda(), enc.attention_mask.cuda()
                lg = used.logits(model, ids, att)
                m = att[:, 1:].float()
                m[:, : n1 - 1] = 0  # score only what follows the label prompt
                tg = used.targets(ids)[..., None]
                for t in temps:
                    tl = torch.log_softmax(lg / t, -1).gather(-1, tg)[..., 0]
                    lp[t][yy, kk, sel] = (tl * m).sum(1).cpu().numpy()
                del ids, att, lg, m, tg
    out = {}
    for t in temps:
        lse = np.logaddexp.reduce(lp[t], axis=1)
        out[t] = lse[1] - lse[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--label_first", action="store_true", help="control: label always first in the fit")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    name = f"any_{a.model.split('/')[-1]}{a.tag}"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tr = pl.read_csv(DATA_DIR / "train.csv", infer_schema=False).with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    og = pl.read_csv(DATA_DIR / "orig" / "credit_card_fraud_2026.csv", infer_schema=False)
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    assert (ref["id"].to_numpy() == tr["id"].to_numpy()).all()
    y = ref[TARGET].to_numpy().astype(int)
    cols = [c for c in tr.columns if c not in ("id", TARGET)]
    y_or = og[TARGET].cast(pl.Float64).cast(pl.Int64).to_numpy()
    f_or = field_texts(og.with_columns(pl.Series(TARGET, y_or.astype(str))), [*cols, TARGET])
    va = np.where(fold_assignment(y) == 1)[0]  # the 30k fold-1 rows replica.py scored (its e10-e40 files)
    rows = np.sort(np.random.default_rng(5).choice(va, a.n, replace=False))
    f_tr = field_texts(tr[rows], cols)
    ys = y[rows]
    print(f"{name}: {len(f_or)} original rows ({y_or.sum()} fraud); eval on {len(rows)} train rows; example:\n  "
          f"{row_texts(f_or[:1], np.random.default_rng(0))[0]}", flush=True)

    te = pl.read_csv(DATA_DIR / "test.csv", infer_schema=False)
    used = Used(tok, [tr, te, og], cols)
    print(f"restricted softmax: {len(used.ids)} tokens", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler()
    per = int(np.ceil(len(f_or) / a.bs))
    total = per * a.epochs
    start = 0
    for e in sorted(a.eval, reverse=True):  # resume from the latest saved epoch
        if (OUT / f"{name}_e{e}.pt").exists():
            st = torch.load(OUT / f"{name}_e{e}.pt", map_location="cuda", weights_only=True)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"])
            start = e
            print(f"resumed at epoch {e}", flush=True)
            break
    rng = np.random.default_rng(start)
    t0, run = time.time(), 0.0
    for ep in range(start, a.epochs):
        model.train()
        perm = rng.permutation(len(f_or))
        for s in range(per):
            step = ep * per + s
            for g in opt.param_groups:
                g["lr"] = a.lr * (1 - step / total)
            sel = perm[s * a.bs : (s + 1) * a.bs]
            if a.label_first:
                texts = [f"{f[-1]}, " + ", ".join(f[:-1][rng.permutation(len(f) - 1)]) for f in f_or[sel]]
            else:
                texts = row_texts(f_or[sel], rng)
            enc = tok(texts, return_tensors="pt", padding=True)
            ids = enc.input_ids.cuda()
            lg = used.logits(model, ids)
            # be_great's collator copies input_ids into labels: the EOS padding is a target too
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), used.targets(ids).reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            run += loss.item() if s % 25 == 0 else 0.0
            del ids, lg, loss
        print(f"epoch {ep + 1}/{a.epochs} loss {run / np.ceil(per / 25):.4f} ({time.time() - t0:.0f}s)", flush=True)
        run = 0.0
        if ep + 1 in a.eval:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict()},
                       OUT / f"{name}_e{ep + 1}.pt")
            torch.cuda.empty_cache()
            t1 = time.time()
            for temp, s_ in bayes_llr(model, tok, used, f_tr, a.k, 128).items():
                print(f"  epoch {ep + 1} T={temp}: AUC {roc_auc_score(ys, s_):.5f}  LLR sd {s_.std():.2f}", flush=True)
                if temp == 1.0:
                    np.save(OUT / f"{name}_e{ep + 1}_f1sub.npy", s_ + np.log(y_or.mean() / (1 - y_or.mean())))
            print(f"  eval {time.time() - t1:.0f}s", flush=True)
            torch.cuda.empty_cache()
    del model, opt, scaler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
