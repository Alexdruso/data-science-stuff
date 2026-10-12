"""PR-35: fine-tuned TabPFN-3 (local v3 checkpoint, no licence) on the exported folds.

    /tmp/tabpfn_venv/bin/python src/run_tabpfn_ft.py --fold 1 --minutes 40 --n-val 20000   # probe
    /tmp/tabpfn_venv/bin/python src/run_tabpfn_ft.py --fold 1 --minutes 40 --save-probe    # full fold

Fine-tuning only sees the fold's training rows (its early-stopping split is carved out of
them), so the validation fold stays honest. Inference context = a stratified subsample of
the training rows, as in run_tabpfn.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier

from run_tabpfn import V3_CKPT, select

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--feats", choices=["raw", "full"], default="full")
    p.add_argument("--minutes", type=float, default=40)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ft-samples", type=int, default=10000, help="ctx+query rows per meta-batch")
    p.add_argument("--ft-pool", type=int, default=200000, help="training rows given to fine-tuning")
    p.add_argument("--ctx", type=int, default=50000, help="inference context rows")
    p.add_argument("--n-est", type=int, default=2)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--tag", default="tabpfn_ft")
    p.add_argument("--save-probe", action="store_true")
    p.add_argument("--test", action="store_true", help="also predict the test set")
    a = p.parse_args()

    k = a.fold
    z, cat = select(dict(np.load(RESULTS / f"tfm_fold{k}.npz", allow_pickle=True)), a.feats)
    rng = np.random.default_rng(42 + k)
    y = z["y_tr"]
    pool = rng.choice(len(y), min(a.ft_pool, len(y)), replace=False)
    clf = FinetunedTabPFNClassifier(
        device="cuda", epochs=a.epochs, time_limit=int(a.minutes * 60), learning_rate=a.lr,
        n_finetune_ctx_plus_query_samples=a.ft_samples, n_inference_subsample_samples=a.ctx,
        n_estimators_finetune=a.n_est, n_estimators_validation=a.n_est,
        n_estimators_final_inference=a.n_est, random_state=42 + k, eval_metric="roc_auc",
        validation_split_ratio=0.05, model_version=ModelVersion.V3,
        extra_classifier_kwargs={"model_path": str(V3_CKPT), "categorical_features_indices": cat,
                                 "ignore_pretraining_limits": True, "memory_saving_mode": True},
    )
    t0 = time.time()
    clf.fit(z["x_tr"][pool], y[pool])
    print(f"fine-tune: {time.time() - t0:.0f}s", flush=True)
    xv = z["x_va"] if a.n_val is None else z["x_va"][: a.n_val]
    yv = z["y_va"][: len(xv)]
    t0 = time.time()
    pv = np.concatenate([clf.predict_proba(xv[i:i + a.chunk])[:, 1] for i in range(0, len(xv), a.chunk)])
    print(f"fold {k}: AUC {roc_auc_score(yv, pv):.5f} ({time.time() - t0:.0f}s predict)", flush=True)
    if a.save_probe:
        np.save(RESULTS / f"probe_{a.tag}_f{k}.npy", np.stack([z["va_idx"][: len(pv)].astype(float), pv]))
    if a.test:
        xt = z["x_te"]
        pt = np.concatenate([clf.predict_proba(xt[i:i + a.chunk])[:, 1] for i in range(0, len(xt), a.chunk)])
        np.save(RESULTS / f"test_{a.tag}_f{k}.npy", pt)
    del clf
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
