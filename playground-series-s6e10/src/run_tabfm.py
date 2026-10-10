"""PR-29: Google TabFM / NVIDIA Kumo via structured-data-models (run with /tmp/sdm_venv/bin/python).

    /tmp/sdm_venv/bin/python src/run_tabfm.py --fold 1 --n-val 20000 --feats full --ctx 8000
    /tmp/sdm_venv/bin/python src/run_tabfm.py --all --bags 4 --tag tabfm_full

The pool's TabFM (cdeotte) runs on the raw columns and earns the stack's largest coefficient;
this one sees our exported fold matrices (route TE fit on the fold's training split, inner
OOF for the context rows). Each bag draws a fresh random context; predictions are the mean
logit over bags. TabFM weights: TabFM Non-Commercial License v1.0.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEV = "cuda"


def table(x: np.ndarray, names: list[str]) -> sdm.TableTensor:
    return sdm.TableTensor.from_pandas(pd.DataFrame(x, columns=names), {n: "numerical" for n in names},
                                       device=DEV)


def predict(model, xq: np.ndarray, names: list[str], qb: int) -> np.ndarray:
    out, i = [], 0
    while i < len(xq):
        try:
            p = model.predict(table(xq[i:i + qb], names)).to_pandas()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            qb = max(16, qb // 2)
            print(f"      OOM -> query batch {qb}", flush=True)
            continue
        out.append(p["yes"].to_numpy(np.float64))
        i += len(p)
    return np.concatenate(out)


def run_bag(model, z: dict, names: list[str], ctx: int, seed: int, queries: list[np.ndarray],
            qb: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(z["y_tr"]), ctx, replace=False))
    y_ctx = sdm.TableTensor.from_pandas(
        pd.DataFrame({"target": np.where(z["y_tr"][idx] == 1, "yes", "no")}), {"target": "categorical"},
        device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    model.fit(x=table(z["x_tr"][idx], names), y=y_ctx, num_estimators=1, estimator_batch_size=1,
              generator=gen)
    out = [predict(model, q, names, qb) for q in queries]
    torch.cuda.empty_cache()
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--bags", type=int, default=1)
    ap.add_argument("--qb", type=int, default=2048)
    ap.add_argument("--feats", choices=["raw", "compact", "full"], default="compact")
    ap.add_argument("--model", choices=["tabfm", "kumo"], default="tabfm")
    ap.add_argument("--tag", default="tabfm")
    ap.add_argument("--test-folds", type=int, nargs="*", default=[1],
                    help="folds whose contexts also predict test (test costs 2x a fold's val rows)")
    a = ap.parse_args()

    # 1.64B params: fp32 does not fit 6 GB and fp16 gives NaN → bf16 weights + bf16 autocast
    model = (sdm.models.TabFM(task="classification", accept_license=True, device="cpu") if a.model == "tabfm"
             else sdm.models.KumoTabular(task="classification", size="large", device="cpu")).eval()
    model = model.to(torch.bfloat16).to(DEV)
    folds = range(1, 6) if a.all else [a.fold]
    oof, test = None, None
    t0 = time.time()
    for k in folds:
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        names = [str(c) for c in z["cols"]]
        if a.feats != "full":
            # compact: raw 21 + route TEs + teacher + original route rate + route count (memory ∝ ctx × cols)
            extra = [] if a.feats == "raw" else [c for c in names if c.startswith("TE_")
                                                  or c in ("og_xgb", "orig_fd_rate", "cnt_flight_distance")]
            keep = list(range(21)) + [names.index(c) for c in extra]
            names = [names[i] for i in keep]
            z = {**z, **{s: z[s][:, keep] for s in ("x_tr", "x_va", "x_te")}}
        for s in ("x_tr", "x_va", "x_te"):
            z[s] = np.nan_to_num(z[s].astype(np.float32), nan=-1.0)
        xv = z["x_va"] if a.n_val is None else z["x_va"][: a.n_val]
        yv = z["y_va"] if a.n_val is None else z["y_va"][: a.n_val]
        do_test = a.all and k in a.test_folds
        queries = [xv] + ([z["x_te"]] if do_test else [])
        acc = [np.zeros(len(q)) for q in queries]
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            for b in range(a.bags):
                for j, p in enumerate(run_bag(model, z, names, a.ctx, 1000 * k + b, queries, a.qb)):
                    acc[j] += logit(p) / a.bags
                print(f"   fold {k} bag {b + 1}: AUC {roc_auc_score(yv, acc[0]):.5f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if a.all:
            if oof is None:
                oof, test = np.zeros(len(z["y_tr"]) + len(z["y_va"])), np.zeros(len(z["x_te"]))
            oof[z["va_idx"]] = acc[0]
            if do_test:
                test += acc[1] / len(a.test_folds)
            np.save(RESULTS / f"partial_{a.tag}.npy", np.stack([oof, np.r_[test, np.zeros(len(oof) - len(test))]]))
        else:
            np.save(RESULTS / f"probe_{a.tag}_f{k}.npy", np.stack([z["va_idx"][: len(xv)].astype(float), acc[0]]))
    if a.all:
        np.save(RESULTS / f"oof_{a.tag}.npy", oof)
        np.save(RESULTS / f"test_{a.tag}.npy", test)
        print(f"saved oof_/test_{a.tag}.npy")


if __name__ == "__main__":
    main()
