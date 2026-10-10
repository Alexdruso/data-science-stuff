"""TabDPT on exported folds (run with /tmp/tabdpt_venv/bin/python, self-contained).

    /tmp/tabdpt_venv/bin/python src/run_tabdpt.py --fold 1 --n-val 20000          # probe
    /tmp/tabdpt_venv/bin/python src/run_tabdpt.py --all --tag tabdpt_full         # 5 folds + test

Retrieval TFM: every query row gets its own context = its nearest training rows
(faiss), so the whole fold's training split is usable — unlike TabPFN/TabICL,
which see a random subsample. Same matrices as run_tabpfn.py (tfm_fold{k}.npz).
Test is predicted once, by the fold-1 model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tabdpt import TabDPTClassifier

RESULTS = Path(__file__).resolve().parents[1] / "results"


def select(z: dict, feats: str) -> dict:
    if feats == "full":
        return z
    return {**z, **{k: z[k][:, :21] for k in ("x_tr", "x_va", "x_te")}}


def predict(model: TabDPTClassifier, x: np.ndarray, ctx: int, bs: int) -> np.ndarray:
    out = [model.predict_proba(x[i:i + 20000], context_size=ctx, batch_size=bs)[:, 1]
           for i in range(0, len(x), 20000)]
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--all", action="store_true")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--feats", choices=["raw", "full"], default="full")
    p.add_argument("--reduction", default="retrieval")
    p.add_argument("--tag", default="tabdpt")
    a = p.parse_args()

    folds = range(1, 6) if a.all else [a.fold]
    oof, test = None, None
    for k in folds:
        z = select(dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)), a.feats)
        t0 = time.time()
        model = TabDPTClassifier(context_reduction=a.reduction, device="cuda", compile=False, use_flash=False,
                                 verbose=False)
        model.fit(z["x_tr"], z["y_tr"])
        xv = z["x_va"] if a.n_val is None else z["x_va"][: a.n_val]
        yv = z["y_va"] if a.n_val is None else z["y_va"][: a.n_val]
        pv = predict(model, xv, a.ctx, a.bs)
        print(f"fold {k}: AUC {roc_auc_score(yv, pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        if a.all:
            if oof is None:
                oof = np.zeros(len(z["y_tr"]) + len(z["y_va"]))
                test = np.zeros(len(z["x_te"]))
            oof[z["va_idx"]] = pv
            if k == 1:
                test = predict(model, z["x_te"], a.ctx, a.bs)
        del model
        torch.cuda.empty_cache()
    if a.all:
        np.save(RESULTS / f"oof_{a.tag}.npy", oof)
        np.save(RESULTS / f"test_{a.tag}.npy", test)
        print(f"saved oof_/test_{a.tag}.npy")


if __name__ == "__main__":
    main()
