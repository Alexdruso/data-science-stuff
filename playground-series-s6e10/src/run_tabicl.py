"""PR-22: TabICL on exported folds (run with /tmp/tabicl_venv/bin/python, self-contained).

    /tmp/tabicl_venv/bin/python src/run_tabicl.py --fold 1 --ctx 50000 --n-val 20000   # timing probe
    /tmp/tabicl_venv/bin/python src/run_tabicl.py --all --ctx 50000                     # 5 folds + test

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
from tabicl import TabICLClassifier

RESULTS = Path(__file__).resolve().parents[1] / "results"


def fit_predict(z: dict, ctx: int, n_est: int, seed: int, n_val: int | None, do_test: bool):
    rng = np.random.default_rng(seed)
    y = z["y_tr"]
    idx = np.concatenate([rng.choice(np.where(y == c)[0], int(round(ctx * (y == c).mean())), replace=False)
                          for c in (0, 1)])
    model = TabICLClassifier(n_estimators=n_est, device="cuda", random_state=seed, batch_size=4)
    model.fit(z["x_tr"][idx], y[idx])
    xv = z["x_va"] if n_val is None else z["x_va"][:n_val]
    t0 = time.time()
    pv = np.concatenate([model.predict_proba(xv[i:i + 20000])[:, 1] for i in range(0, len(xv), 20000)])
    print(f"   val predict {len(xv)} rows: {time.time() - t0:.0f}s", flush=True)
    pt = None
    if do_test:
        xt = z["x_te"]
        pt = np.concatenate([model.predict_proba(xt[i:i + 20000])[:, 1] for i in range(0, len(xt), 20000)])
    del model
    torch.cuda.empty_cache()
    return pv, pt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--all", action="store_true")
    p.add_argument("--ctx", type=int, default=50000)
    p.add_argument("--n-est", type=int, default=4)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--tag", default="tabicl")
    a = p.parse_args()

    folds = range(1, 6) if a.all else [a.fold]
    oof, test = None, None
    for k in folds:
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        t0 = time.time()
        # Test is predicted ONCE, by the fold-1 model: 300k in-context predictions per fold
        # would cost 5x the val predictions (memory: reference_tabpfn3_setup).
        pv, pt = fit_predict(z, a.ctx, a.n_est, 42 + k, a.n_val, a.all and k == 1)
        yv = z["y_va"] if a.n_val is None else z["y_va"][: a.n_val]
        print(f"fold {k}: AUC {roc_auc_score(yv, pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
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
