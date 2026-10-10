"""PR-26: TabPFN-3 on exported folds (run with /tmp/tabpfn_venv/bin/python, self-contained).

    /tmp/tabpfn_venv/bin/python src/run_tabpfn.py --fold 1 --ctx 50000 --n-val 20000   # timing probe
    /tmp/tabpfn_venv/bin/python src/run_tabpfn.py --all --ctx 50000                     # 5 folds + test

Context = a stratified random subsample of the fold's training rows (in-context
learning cannot take 560k rows on 6 GB). Saves raw val/test probabilities; the
gate runs afterwards in the main venv. Test is predicted once, by the fold-1 model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHUNK = 5000


V3_CKPT = Path.home() / ".cache/tabpfn/tabpfn-v3-classifier-v3_default.ckpt"


def select(z: dict, feats: str) -> tuple[dict, list[int]]:
    """raw: the 21 raw columns with flight_distance categorical; full: all 69 exported columns."""
    cols = list(z["cols"])
    cat = [int(i) for i in z["cat_idx"]]
    if feats == "full":
        return z, cat
    keep = list(range(21))
    z = {**z, **{k: z[k][:, keep] for k in ("x_tr", "x_va", "x_te")}}
    return z, sorted({*cat, cols.index("flight_distance")})


def fit_predict(z: dict, ctx: int, n_est: int, seed: int, n_val: int | None, do_test: bool,
                cat: list[int], model_path: str):
    rng = np.random.default_rng(seed)
    y = z["y_tr"]
    idx = np.concatenate([rng.choice(np.where(y == c)[0], int(round(ctx * (y == c).mean())), replace=False)
                          for c in (0, 1)])
    kw = {"model_path": model_path} if model_path else {}
    model = TabPFNClassifier(n_estimators=n_est, device="cuda", random_state=seed,
                             ignore_pretraining_limits=True, categorical_features_indices=cat,
                             memory_saving_mode=True, **kw)
    model.fit(z["x_tr"][idx], y[idx])
    xv = z["x_va"] if n_val is None else z["x_va"][:n_val]
    t0 = time.time()
    pv = np.concatenate([model.predict_proba(xv[i:i + CHUNK])[:, 1] for i in range(0, len(xv), CHUNK)])
    print(f"   val predict {len(xv)} rows: {time.time() - t0:.0f}s", flush=True)
    pt = None
    if do_test:
        xt = z["x_te"]
        pt = np.concatenate([model.predict_proba(xt[i:i + CHUNK])[:, 1] for i in range(0, len(xt), CHUNK)])
    del model
    torch.cuda.empty_cache()
    return pv, pt


def local_predict(z: dict, xq: np.ndarray, ctx: int, n_est: int, seed: int, cat: list[int],
                  model_path: str, batch: int, local_frac: float) -> np.ndarray:
    """Route-local context: queries sorted by flight_distance, batch by batch; each batch's
    context = training rows on the batch's routes (≤ local_frac·ctx) + a random fill."""
    rng = np.random.default_rng(seed)
    fd = list(z["cols"]).index("flight_distance")
    x_tr, y = z["x_tr"], z["y_tr"]
    order = np.argsort(xq[:, fd], kind="stable")
    by_fd: dict[float, np.ndarray] = {}
    srt = np.argsort(x_tr[:, fd], kind="stable")
    vals, starts = np.unique(x_tr[srt, fd], return_index=True)
    for v, a, b in zip(vals, starts, [*starts[1:], len(srt)]):
        by_fd[float(v)] = srt[a:b]
    kw = {"model_path": model_path} if model_path else {}
    out = np.zeros(len(xq))
    t0 = time.time()
    for i in range(0, len(xq), batch):
        q = order[i:i + batch]
        loc = np.concatenate([by_fd.get(float(v), np.empty(0, int)) for v in np.unique(xq[q, fd])])
        cap = int(ctx * local_frac)
        if len(loc) > cap:
            loc = rng.choice(loc, cap, replace=False)
        rest = np.setdiff1d(rng.choice(len(y), ctx - len(loc) + 5000, replace=False), loc)[: ctx - len(loc)]
        idx = np.concatenate([loc, rest])
        model = TabPFNClassifier(n_estimators=n_est, device="cuda", random_state=seed,
                                 ignore_pretraining_limits=True, categorical_features_indices=cat,
                                 memory_saving_mode=True, **kw)
        model.fit(x_tr[idx], y[idx])
        out[q] = model.predict_proba(xq[q])[:, 1]
        del model
        torch.cuda.empty_cache()
        if i == 0:
            print(f"   batch 1: {len(loc)} local ctx rows, {time.time() - t0:.0f}s", flush=True)
    print(f"   local predict {len(xq)} rows: {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--all", action="store_true")
    p.add_argument("--ctx", type=int, default=50000)
    p.add_argument("--n-est", type=int, default=4)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--tag", default="tabpfn")
    p.add_argument("--feats", choices=["raw", "full"], default="full")
    p.add_argument("--model-path", default=str(V3_CKPT), help="'' = default (3.5, licence-gated)")
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--save-probe", action="store_true", help="save the probe's val preds")
    p.add_argument("--local", type=float, default=0.0, help="route-local context fraction (0 = off)")
    p.add_argument("--batch", type=int, default=5000)
    a = p.parse_args()
    global CHUNK
    CHUNK = a.chunk

    folds = range(1, 6) if a.all else [a.fold]
    oof, test = None, None
    for k in folds:
        z, cat = select(dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)), a.feats)
        t0 = time.time()
        # Test is predicted ONCE, by the fold-1 model: 300k in-context predictions per fold
        # would cost 5x the val predictions (memory: reference_tabpfn3_setup).
        if a.local > 0:
            xv = z["x_va"] if a.n_val is None else z["x_va"][: a.n_val]
            t0 = time.time()
            pv = local_predict(z, xv, a.ctx, a.n_est, 42 + k, cat, a.model_path, a.batch, a.local)
            pt = (local_predict(z, z["x_te"], a.ctx, a.n_est, 42 + k, cat, a.model_path, a.batch, a.local)
                  if a.all and k == 1 else None)
        else:
            pv, pt = fit_predict(z, a.ctx, a.n_est, 42 + k, a.n_val, a.all and k == 1, cat, a.model_path)
        yv = z["y_va"] if a.n_val is None else z["y_va"][: a.n_val]
        print(f"fold {k}: AUC {roc_auc_score(yv, pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        if a.save_probe and not a.all:
            n = len(pv)
            np.save(RESULTS / f"probe_{a.tag}_f{k}.npy", np.stack([z["va_idx"][:n].astype(float), pv]))
        if a.all:
            if oof is None:
                oof = np.zeros(len(z["y_tr"]) + len(z["y_va"]))
                test = np.zeros(len(z["x_te"]))
            oof[z["va_idx"]] = pv
            if pt is not None:
                test = pt
    if a.all:
        np.save(RESULTS / f"oof_{a.tag}.npy", oof)
        np.save(RESULTS / f"test_{a.tag}.npy", test)
        print(f"saved oof_/test_{a.tag}.npy")


if __name__ == "__main__":
    main()
