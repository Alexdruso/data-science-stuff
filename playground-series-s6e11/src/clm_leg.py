"""Compact-format GPT-2 leg with the label as a field: label-first, label-last or label-anywhere.

A row is BOS + 25 fields in random order. A field is one learned column token followed by the
GPT-2 tokens of its value; the label is the 25th field (column token LABQ, value LAB0/LAB1).
Targets: the value tokens, END after a value's last token, and the label token after LABQ.
Column tokens are inputs only. The softmax covers the used value tokens + END + 2 labels.
(Compact format after megayak's public S6E11 write-up; label-anywhere after Deotte's S6E10.)

`--label first|last|any` sets where the label field sits during the fit. One fitted model is
read three ways, each a per-row log-odds (mean over K draws of order):
    first  log p(y | BOS)        + sum over all fields of [log p(x | 1) - log p(x | 0)]   (Storyteller)
    last   log p(y | all fields)                                                           (Detective)
    any    log p(y | prefix)     + suffix LLR, label at a uniform random position          (chain)

    ~/venvs/lm/bin/python src/clm_leg.py --folds 1 --label any --epochs 10 --tag _any_ep10 --no_test
Outputs: results/_clm<tag>_{first,last,any}/f{k}_{val,test}.npy (log-odds), f{k}_model.pt in _first.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
DEV = "cuda"
READS = ("first", "last", "any")


class Table:
    """All rows (train, test, orig) as int16 [n, 25, W] token grids on the GPU. Field 24 = label."""

    def __init__(self, frames: list[pl.DataFrame], cols: list[str], tok: AutoTokenizer) -> None:
        allv = pl.concat([f.select(cols) for f in frames], how="vertical")
        per, used = [], set()
        for c in cols:
            vals = allv[c].to_numpy()
            uniq, inv = np.unique(vals, return_inverse=True)
            ids = [tok.encode(" " + (str(v).strip() or "None")) for v in uniq]
            used.update(i for x in ids for i in x)
            per.append((inv, ids))
        self.vocab = np.array(sorted(used))
        remap = {g: i for i, g in enumerate(self.vocab)}
        v = len(self.vocab)
        self.V, self.END, self.LAB, self.PAD = v, v, v + 1, v + 3  # outputs: [0, V+3)
        self.COL0 = v + 4
        self.NC = len(cols) + 1
        self.BOS = self.COL0 + self.NC
        self.NTOK = self.BOS + 1
        self.W = 1 + max(len(x) for _, ids in per for x in ids)
        grid = np.full((len(allv), self.NC, self.W), -1, np.int16)
        for j, (inv, ids) in enumerate(per):
            t = np.full((len(ids), self.W), -1, np.int16)
            t[:, 0] = self.COL0 + j
            for i, x in enumerate(ids):
                t[i, 1 : 1 + len(x)] = [remap[g] for g in x]
            grid[:, j, :] = t[inv]
        grid[:, -1, 0] = self.COL0 + len(cols)  # LABQ; the label value is written per batch
        self.grid = torch.from_numpy(grid).to(DEV)
        self.lens = (grid >= 0).sum((1, 2)) + 2  # + BOS + label value
        self.T = int(self.lens.max())
        self.col_names = [tok.encode(" " + c) for c in cols] + [tok.encode(" is_fraud")]
        self.lab_src = [tok.encode(" False")[0], tok.encode(" True")[0]]
        self.comma = tok.encode(",")[0]

    def assemble(self, idx: torch.Tensor, lab: torch.Tensor, perm: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inputs [B, L], targets [B, L] (-100 = not scored), position of LABQ [B]."""
        b = idx.numel()
        seg = self.grid[idx].long()
        seg[:, -1, 1] = self.LAB + lab
        seg = seg.gather(1, perm[:, :, None].expand(-1, -1, self.W)).reshape(b, -1)
        keep = torch.argsort((seg < 0).to(torch.int8), dim=1, stable=True)
        seg = seg.gather(1, keep)
        n = int((seg >= 0).sum(1).max())
        seq = torch.cat([torch.full((b, 1), self.BOS, device=DEV), seg[:, :n]], 1)
        seq = torch.where(seq < 0, torch.full_like(seq, self.PAD), seq)
        cur, nxt = seq[:, :-1], seq[:, 1:]
        is_val = (nxt < self.V) | (nxt == self.LAB) | (nxt == self.LAB + 1)
        tgt = torch.where(is_val, nxt, torch.full_like(nxt, self.END))
        tgt = torch.where((cur == self.PAD) | (cur == self.BOS), torch.full_like(tgt, -100), tgt)
        labq = (cur == self.COL0 + self.NC - 1).float().argmax(1)
        return cur.contiguous(), tgt, labq


def build_model(name: str, tb: Table) -> torch.nn.Module:
    """Pretrained GPT-2 body with a compact embedding table initialised from the GPT-2 rows."""
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    w = model.transformer.wte.weight.data
    emb = torch.nn.Embedding(tb.NTOK, w.shape[1])
    with torch.no_grad():
        emb.weight[: tb.V] = w[torch.from_numpy(tb.vocab)]
        emb.weight[tb.END] = w[tb.comma]
        emb.weight[tb.LAB] = w[tb.lab_src[0]]
        emb.weight[tb.LAB + 1] = w[tb.lab_src[1]]
        emb.weight[tb.PAD] = w.mean(0)
        emb.weight[tb.BOS] = w[50256]
        for j, ids in enumerate(tb.col_names):
            emb.weight[tb.COL0 + j] = w[torch.tensor(ids)].mean(0)
    model.transformer.wte = emb
    model.lm_head = torch.nn.Identity()
    return model.to(DEV)


def logits(model: torch.nn.Module, tb: Table, x: torch.Tensor) -> torch.Tensor:
    """fp16 body, fp32 output layer over the V + 3 output tokens."""
    with torch.autocast("cuda", dtype=torch.float16):
        h = model.transformer(input_ids=x).last_hidden_state
    return h.float() @ model.transformer.wte.weight[: tb.V + 3].T


def field_perm(b: int, nc: int, mode: str, gen: torch.Generator) -> torch.Tensor:
    """Random field order; the label field (index nc-1) first, last, or at a uniform position."""
    feat = torch.rand(b, nc - 1, device=DEV, generator=gen).argsort(1)
    lab = torch.full((b, 1), nc - 1, device=DEV)
    if mode == "first":
        return torch.cat([lab, feat], 1)
    if mode == "last":
        return torch.cat([feat, lab], 1)
    return torch.rand(b, nc, device=DEV, generator=gen).argsort(1)


def length_batches(lens: np.ndarray, rows: np.ndarray, bs: int, rs: np.random.RandomState,
                   group: int = 64) -> np.ndarray:
    """Random batches of similar length (sorted inside chunks of `group` batches)."""
    o = rs.permutation(rows)[: len(rows) // bs * bs]
    for c in range(0, len(o), bs * group):
        s = o[c : c + bs * group]
        o[c : c + bs * group] = s[np.argsort(lens[s], kind="stable")]
    return o.reshape(-1, bs)[rs.permutation(len(o) // bs)]


def fit(model: torch.nn.Module, tb: Table, rows: np.ndarray, y_dev: torch.Tensor, mode: str,
        epochs: int, lr: float, bs: int, seed: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler()
    per = len(rows) // bs
    total = per * epochs
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rs = np.random.RandomState(seed)
    model.train()
    t0, run, step = time.time(), 0.0, 0
    print(f"  fit {len(rows)} rows, label {mode}, {total} steps", flush=True)
    for _ in range(epochs):
        bt = torch.as_tensor(length_batches(tb.lens, rows, bs, rs), device=DEV)
        for s in range(per):
            for grp in opt.param_groups:
                grp["lr"] = lr * min(1.0, (step + 1) / 200) * max(0.0, 1 - step / total)
            idx = bt[s]
            x, tgt, _ = tb.assemble(idx, y_dev[idx], field_perm(bs, tb.NC, mode, gen))
            lg = logits(model, tb, x)
            loss = F.cross_entropy(lg.view(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            run += loss.item() if (step + 1) % 50 == 0 else 0.0
            step += 1
            if step % 2000 == 0:
                print(f"  step {step}/{total} loss {run / 40:.4f} {step * bs / (time.time() - t0):.0f} rows/s",
                      flush=True)
                run = 0.0
            del x, tgt, lg, loss
    del opt, scaler


@torch.no_grad()
def score(model: torch.nn.Module, tb: Table, rows: np.ndarray, mode: str, k: int, seed: int,
          bs: int = 256) -> np.ndarray:
    """Per-row log-odds for one read: log p(y=1|prefix) - log p(y=0|prefix) + suffix LLR; mean over k."""
    model.eval()
    srt = np.argsort(tb.lens[rows], kind="stable")
    rows_t = torch.as_tensor(rows[srt], device=DEV)
    out = torch.zeros(len(rows), device=DEV, dtype=torch.float64)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(k):
        for s in range(0, len(rows), bs):
            idx = rows_t[s : s + bs]
            n = idx.numel()
            perm = field_perm(n, tb.NC, mode, gen)
            x0, tgt, labq = tb.assemble(idx, torch.zeros(n, dtype=torch.long, device=DEV), perm)
            ar = torch.arange(n, device=DEV)
            x1 = x0.clone()
            has = labq + 1 < x0.shape[1]  # label value is an input unless it is the very last token
            x1[ar[has], labq[has] + 1] = tb.LAB + 1
            lp = torch.log_softmax(logits(model, tb, torch.cat([x0, x1])), -1)
            head = lp[ar, labq, tb.LAB + 1] - lp[ar, labq, tb.LAB]
            after = torch.arange(x0.shape[1], device=DEV)[None, :] > labq[:, None] + 1
            m = (after & (tgt >= 0)).double()  # tokens after the label value
            g = tgt.clamp_min(0)[..., None]
            ll = (lp[n:].gather(-1, g)[..., 0] - lp[:n].gather(-1, g)[..., 0]).double()
            out[s : s + n] += (head.double() + (ll * m).sum(1)) / k
            del x0, x1, tgt, lp, ll, m
    res = np.empty(len(rows))
    res[srt] = out.cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--label", default="any", choices=READS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--neg_frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k_val", type=int, default=4)
    ap.add_argument("--k_test", type=int, default=2)
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="stop the fit after N steps (speed check)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    reads = READS if a.label == "any" else (a.label,)
    outs = {r: RESULTS_DIR / f"_clm{a.tag}_{r}" for r in reads}
    for o in outs.values():
        o.mkdir(exist_ok=True)
    mdir = outs[reads[0]]

    tok = AutoTokenizer.from_pretrained("gpt2")
    rd = lambda p: pl.read_csv(p, infer_schema=False)  # noqa: E731
    tr = rd(DATA_DIR / "train.csv").with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    te = rd(DATA_DIR / "test.csv").with_columns(pl.col("id").cast(pl.Int64)).sort("id")
    og = rd(DATA_DIR / "orig" / "credit_card_fraud_2026.csv")
    ref = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    assert (ref["id"].to_numpy() == tr["id"].to_numpy()).all()
    y = ref[TARGET].to_numpy().astype(int)
    folds = fold_assignment(y)
    cols = [c for c in tr.columns if c not in ("id", TARGET)]
    t0 = time.time()
    tb = Table([tr, te, og], cols, tok)
    ntr, nte = len(tr), len(te)
    print(f"table {tuple(tb.grid.shape)} V {tb.V} T {tb.T} mean len {tb.lens[:ntr].mean():.1f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    y_dev = torch.zeros(len(tb.lens), dtype=torch.long, device=DEV)
    y_dev[:ntr] = torch.as_tensor(y, device=DEV)

    for k in a.folds:
        if all((outs[r] / f"f{k}_val.npy").exists() for r in reads):
            continue
        rng = np.random.default_rng(1000 + k)
        tr_idx, va_idx = np.where(folds != k)[0], np.where(folds == k)[0]
        keep = (y[tr_idx] == 1) | (rng.random(len(tr_idx)) < a.neg_frac)
        fit_idx = tr_idx[keep]
        # the fit sees neg_frac of the negatives: every read's label head is off by log(neg_frac)
        prior_fix = np.log(a.neg_frac)
        torch.manual_seed(1000 + k)
        model = build_model(a.model, tb)
        print(f"fold {k}:", flush=True)
        if a.bench:
            t1 = time.time()
            fit(model, tb, fit_idx[: a.bench * a.bs], y_dev, a.label, 1, a.lr, a.bs, 1000 + k)
            torch.cuda.synchronize()
            print(f"  bench {a.bench * a.bs / (time.time() - t1):.0f} rows/s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            t1 = time.time()
            sv = score(model, tb, va_idx[:20000], "first", 1, 0)
            print(f"  score {20000 / (time.time() - t1):.0f} rows/s; auc {roc_auc_score(y[va_idx[:20000]], sv):.4f}",
                  flush=True)
            del model
            torch.cuda.empty_cache()
            return
        fit(model, tb, fit_idx, y_dev, a.label, a.epochs, a.lr, a.bs, 1000 + k)
        torch.save(model.state_dict(), mdir / f"f{k}_model.pt")
        for r in reads:
            t1 = time.time()
            sv = score(model, tb, va_idx, r, a.k_val, k) + prior_fix
            np.save(outs[r] / f"f{k}_val.npy", sv)
            print(f"  fold {k} read {r}: val auc {roc_auc_score(y[va_idx], sv):.5f} ({time.time() - t1:.0f}s)",
                  flush=True)
            if not a.no_test:
                st = score(model, tb, np.arange(ntr, ntr + nte), r, a.k_test, 100 + k) + prior_fix
                np.save(outs[r] / f"f{k}_test.npy", st)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
