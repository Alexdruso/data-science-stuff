"""PR-36: new-family legs from pytabkit on the exported fold-safe matrices (tfm_fold{k}.npz).

    python src/pytabkit_leg.py --model tabr --folds 1          # fold-1 probe (saves val + test)
    python src/pytabkit_leg.py --model xrfm --folds 1 2 3 4 5

Families absent from the public pool: RealTabR (retrieval-augmented NN: learned k-NN attention
over training rows) and xRFM (recursive feature machine: kernel regression with a learned
feature metric, tree-partitioned). pytabkit early-stops on its own split of the TRAINING rows,
so the scored validation fold never selects anything (no epoch-selection inflation).
Per-fold outputs: results/_pt_<model>/f{k}_{val,test}.npy (resumable).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _chunk_xrfm_soft_predict() -> None:
    """xRFM's soft-tree prediction materialises (rows x leaves) tensors -> OOM on 6 GB for
    100k+ rows; evaluate it in row chunks (identical result)."""
    from xrfm import xrfm as xm

    orig = xm.xRFM._predict_tree_soft
    if getattr(orig, "_chunked", False):
        return

    def chunked(self, X, tree, proba=False):
        n = 8192
        if X.shape[0] <= n:
            return orig(self, X, tree, proba=proba)
        return torch.cat([orig(self, X[i:i + n], tree, proba=proba) for i in range(0, X.shape[0], n)])

    chunked._chunked = True
    xm.xRFM._predict_tree_soft = chunked


def make(model: str, seed: int, n_tr: int):
    if model == "tabr":
        from pytabkit import RealTabR_D_Classifier

        return RealTabR_D_Classifier(
            device="cuda", random_state=seed, val_metric_name="cross_entropy", batch_size=512,
            eval_batch_size=1024, candidate_encoding_batch_size=2048, memory_efficient=True,
            freeze_contexts_after_n_epochs=3, verbosity=1)
    if model == "xrfm":
        from pytabkit import XRFM_D_Classifier

        _chunk_xrfm_soft_predict()

        return XRFM_D_Classifier(device="cuda", random_state=seed, val_metric_name="cross_entropy",
                                 max_leaf_samples=int(os.environ.get("XRFM_LEAF", 8000)),
                                 M_batch_size=2000, verbosity=1)
    if model == "ftt":
        from pytabkit import FTT_D_Classifier

        # RTDL defaults (300 epochs, patience 16, bs 256) take days at 560k rows on a 2060
        return FTT_D_Classifier(device="cuda", random_state=seed, val_metric_name="cross_entropy",
                                batch_size=1024, lr=3e-4,
                                max_epochs=int(os.environ.get("FTT_EPOCHS", 30)),
                                es_patience=int(os.environ.get("FTT_PATIENCE", 4)), verbosity=1)
    if model == "resnet":
        from pytabkit import Resnet_RTDL_D_Classifier

        return Resnet_RTDL_D_Classifier(device="cuda", random_state=seed, val_metric_name="cross_entropy",
                                        batch_size=1024, verbosity=1)
    raise ValueError(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[1])
    p.add_argument("--n-train", type=int, default=0, help="subsample training rows (0 = all)")
    p.add_argument("--no-test", action="store_true")
    a = p.parse_args()
    out = RESULTS / f"_pt_{a.model}{'_n' + str(a.n_train) if a.n_train else ''}"
    out.mkdir(exist_ok=True)
    for k in a.folds:
        if (out / f"f{k}_val.npy").exists():
            print(f"fold {k}: done already")
            continue
        z = dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True))
        cols = [str(c) for c in z["cols"]]
        # TabR's categorical path breaks on this sklearn (imputer drops the columns) -> numeric codes
        cat = set() if a.model == "tabr" else set(int(i) for i in z["cat_idx"])

        def frame(x: np.ndarray) -> pd.DataFrame:
            df = pd.DataFrame(x, columns=cols)
            for i in cat:
                df[cols[i]] = df[cols[i]].astype(int).astype("category")
            return df

        med = np.nanmedian(z["x_tr"], axis=0)  # training-split medians only
        for key in ("x_tr", "x_va", "x_te"):
            x = z[key].astype(np.float32)
            r, c = np.where(np.isnan(x))
            x[r, c] = med[c]
            z[key] = x
        xtr, ytr = z["x_tr"], z["y_tr"].astype(int)
        if a.n_train:
            idx = np.random.default_rng(k).choice(len(ytr), a.n_train, replace=False)
            xtr, ytr = xtr[idx], ytr[idx]
        t0 = time.time()
        m = make(a.model, 42 + k, len(ytr))
        m.fit(frame(xtr), ytr)
        pv = m.predict_proba(frame(z["x_va"]))[:, 1]
        print(f"fold {k}: AUC {roc_auc_score(z['y_va'], pv):.5f} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out / f"f{k}_val.npy", np.stack([z["va_idx"].astype(float), pv]))
        if not a.no_test:
            np.save(out / f"f{k}_test.npy", m.predict_proba(frame(z["x_te"]))[:, 1])
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
