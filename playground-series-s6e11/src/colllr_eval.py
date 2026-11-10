"""Fold-level read: do per-column LM LLRs add to the pooled stack beyond the LM total?

    python src/colllr_eval.py _lm_distilgpt2_ep10 --fold 1 --ext ext/dl/... --ours lm_distilgpt2_ep10

Rows = the fold's val rows (each scored by a model that never saw them). Members enter as
logit(rank) computed on the full OOF; inner 5-fold LR stack on the fold's rows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_COLS, TARGET, build_features

from data_science_stuff.kaggle.cv import fold_assignment
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.predictions import load_predictions
from data_science_stuff.kaggle.stacking import load_oof_banks, logit, rank_logit

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--ext", nargs="*", default=[])
    ap.add_argument("--ours", nargs="*", default=[])
    ap.add_argument("--min_auc", type=float, default=0.946)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--base_stack", default=None, help="use one saved stack's logit as the base")
    a = ap.parse_args()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    n_te = 299984
    m = fold_assignment(y) == a.fold
    mem = load_oof_banks(a.ext, n_train=len(y), n_test=n_te)
    for n in a.ours:
        mem[n] = (np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel(), np.zeros(n_te))
    names = [k for k, v in mem.items() if roc_auc_score(y, v[0]) >= a.min_auc]
    base = np.column_stack([rank_logit(mem[k][0])[m] for k in names]) if names else None
    if a.base_stack:
        p0 = np.clip(np.asarray(load_predictions(RESULTS_DIR, a.base_stack, ids)).ravel(), 1e-12,
                     1 - 1e-12)
        base = logit(p0)[m][:, None]
        names = [a.base_stack]
    ys = y[m]
    d = RESULTS_DIR / a.dir
    llr = np.load(d / f"f{a.fold}_val_colllr.npy")
    tot = np.load(d / f"f{a.fold}_val_coltot.npy")
    lse = np.logaddexp.reduce(tot.astype(float), axis=1)
    lm_lse = lse[1] - lse[0]
    lm_mean = llr.sum(1)
    print(f"{len(names)} members; fold {a.fold}: {m.sum()} rows")
    print(f"colscore solo: lse {roc_auc_score(ys, lm_lse):.5f}  mean {roc_auc_score(ys, lm_mean):.5f}")
    for j in np.argsort(-np.abs(llr).mean(0)):
        print(f"   {RAW_COLS[j]:30s} |llr| {np.abs(llr[:, j]).mean():.3f}  "
              f"solo auc {roc_auc_score(ys, llr[:, j]):.4f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def stack(x: np.ndarray, C: float = a.C) -> float:
        x = StandardScaler().fit_transform(x)
        p = cross_val_predict(LogisticRegression(C=C, max_iter=5000), x, ys, cv=cv,
                              method="decision_function")
        return float(roc_auc_score(ys, p))

    b0 = stack(base)
    print(f"base stack            {b0:.6f}")
    print(f"+ LM lse total        {stack(np.column_stack([base, lm_lse])) - b0:+.6f}")
    print(f"+ 24 col LLRs         {stack(np.column_stack([base, llr])) - b0:+.6f}")
    print(f"+ lse + 24 col LLRs   {stack(np.column_stack([base, lm_lse, llr])) - b0:+.6f}")
    for C in (0.01, 0.1):
        print(f"  C={C}: + lse {stack(np.column_stack([base, lm_lse]), C) - stack(base, C):+.6f}"
              f"  + lse + cols {stack(np.column_stack([base, lm_lse, llr]), C) - stack(base, C):+.6f}")


if __name__ == "__main__":
    main()
