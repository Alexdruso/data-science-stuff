"""PR-17 read: an LM pilot scored on a fold-val subset vs stored LMs, against pool_v7, same rows.

    python src/lm_sub_pool_eval.py _lm_SmolLM2-135M_ep3 _lm_SmolLM2-135M_ep5 [--fold 1]

Refs (full-fold f{k}_val.npy, ordered like the fold's val rows): ep10, ep5, ep20 GPT-2 LMs.
Gain = inner-CV (5 folds x 3 seeds) AUC of LR([pool logit, LM]) minus LR([pool logit]).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)
REFS = ["_lm_distilgpt2_ep5", "_lm_distilgpt2_ep10", "_lm_distilgpt2_ep20"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--base", default="pool_v7")
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    va = np.where(fold_assignment(y) == a.fold)[0]
    pos = {r: i for i, r in enumerate(va)}
    idx = np.load(RESULTS_DIR / a.dirs[0] / f"f{a.fold}_valsub_idx.npy")
    for d in a.dirs[1:]:
        assert (np.load(RESULTS_DIR / d / f"f{a.fold}_valsub_idx.npy") == idx).all(), d
    j = np.array([pos[r] for r in idx])
    ys = y[idx]
    p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.base, ids)).ravel(), 1e-12, 1 - 1e-12)
    base = logit(p0)[idx][:, None]
    masa = np.load("ext/dl/s6e11-gpt2-target-first-oof/oof.npy")[idx]

    def stack(x: np.ndarray) -> float:
        out = []
        for s in (0, 1, 2):
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            p = cross_val_predict(LogisticRegression(max_iter=3000), x, ys, cv=cv,
                                  method="decision_function")
            out.append(roc_auc_score(ys, p))
        return float(np.mean(out))

    b0 = stack(base)
    print(f"{len(idx)} rows, {ys.sum()} pos; base {b0:.6f}")
    cols = {r: np.load(RESULTS_DIR / r / f"f{a.fold}_val.npy")[j] for r in REFS}
    cols |= {d: np.load(RESULTS_DIR / d / f"f{a.fold}_valsub.npy") for d in a.dirs}
    for name, v in cols.items():
        g = stack(np.column_stack([base, v])) - b0
        print(f"  {name:28s} solo {roc_auc_score(ys, v):.5f}  gain {g:+.6f}  "
              f"rho(masa) {np.corrcoef(v, masa)[0, 1]:.4f}  rho(ep10) "
              f"{np.corrcoef(v, cols['_lm_distilgpt2_ep10'])[0, 1]:.4f}")


if __name__ == "__main__":
    main()
