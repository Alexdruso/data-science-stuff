"""Pooled stack over every same-fold public OOF bank + our legs.

    python src/pool_stack.py --ext ext/dl ext/ko ... --ours lm_distilgpt2_ep10 [--out name]

Bank discovery (recursive): `<base>_oof.npy` with `<base>_{full,test}.npy`, or a directory
holding `oof.npy` + `test.npy` (masayakawamata format), or `oof_*.npy`/`test_*.npy` pairs.
Arrays must be n_train / n_test long, in id-sorted (= csv) order. Near-duplicates (rank corr
> --dedup) keep the higher-AUC member. Members enter as logit(rank); meta = L2 LR; the stack
score is outer StratifiedKFold(5, seed 7/13/99), which is nested w.r.t. the meta fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.predictions import load_predictions, save_predictions
from data_science_stuff.kaggle.stacking import load_oof_banks, logit, nested_lr_stack, rank_logit

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def tf_raw(v: np.ndarray) -> np.ndarray:
    """Member score as a standardized logit (log-odds kept, probabilities logit-transformed)."""
    if v.min() >= 0 and v.max() <= 1:
        v = logit(v, eps=1e-7)
    return (v - v.mean()) / (v.std() + 1e-12)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", nargs="*", default=["ext"])
    ap.add_argument("--ours", nargs="*", default=["lm_distilgpt2_ep10"])
    ap.add_argument("--min_auc", type=float, default=0.946)
    ap.add_argument("--dedup", type=float, default=0.9995)
    ap.add_argument("--C", type=float, nargs="+", default=[1.0])
    ap.add_argument("--drop", nargs="*", default=[], help="substrings of member names to drop")
    ap.add_argument("--ablate", nargs="*", default=[], help="substrings; report stack without each")
    ap.add_argument("--raw", action="store_true", help="standardized logits instead of rank-logits")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


class Pool(NamedTuple):
    kept: list[str]
    Xo: np.ndarray
    Xt: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    te_ids: np.ndarray


def build_pool(a: argparse.Namespace) -> Pool:
    """Load banks + our legs, filter by AUC, dedup by corr, return the member matrices."""
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
    ids, y = tr["id"].to_numpy(), tr[TARGET].to_numpy().astype(int)
    assert (np.diff(ids) > 0).all()
    mem = load_oof_banks(a.ext, n_train=len(y), n_test=len(te_ids))
    for n in a.ours:
        mem[n] = (np.asarray(load_predictions(RESULTS_DIR, n, ids)).ravel(),
                  np.asarray(load_predictions(RESULTS_DIR, n, te_ids, split="test")).ravel())
    mem = {k: v for k, v in mem.items() if not any(s in k for s in a.drop)}
    auc = {k: roc_auc_score(y, v[0]) for k, v in mem.items()}
    names = [k for k in sorted(auc, key=lambda k: -auc[k]) if auc[k] >= a.min_auc]
    tf = tf_raw if a.raw else rank_logit
    R = {k: tf(mem[k][0]) for k in names}
    kept: list[str] = []
    for k in names:
        dup = next((j for j in kept if np.corrcoef(R[k], R[j])[0, 1] > a.dedup), None)
        if dup is None:
            kept.append(k)
    print(f"{len(mem)} loaded, {len(names)} >= {a.min_auc}, {len(kept)} after dedup")
    for k in kept:
        print(f"  {k:60s} {auc[k]:.5f}")
    Xo = np.column_stack([R[k] for k in kept])
    Xt = np.column_stack([tf(mem[k][1]) for k in kept])
    return Pool(kept, Xo, Xt, y, ids, te_ids)


def main() -> None:
    a = parse_args()
    kept, Xo, Xt, y, ids, te_ids = build_pool(a)

    def stack(cols: list[int], C: float) -> float:
        return nested_lr_stack(Xo, y, c=C, cols=cols).auc

    allc = list(range(len(kept)))
    best = (-1.0, 0.0)
    for C in a.C:
        s = stack(allc, C)
        print(f"ALL C={C:<8} {s:.6f}")
        best = max(best, (s, C))
    C = best[1]
    ours = [i for i, k in enumerate(kept) if ":" not in k and k in a.ours]
    if ours:
        print(f"without ours ({[kept[i] for i in ours]}): "
              f"{stack([i for i in allc if i not in ours], C):.6f}")
    for sub in a.ablate:
        cols = [i for i, k in enumerate(kept) if sub not in k]
        print(f"without *{sub}* ({len(allc) - len(cols)} members): {stack(cols, C):.6f}")
    full_auc, oof, tst = nested_lr_stack(Xo, y, Xt, c=C)
    full = LogisticRegression(C=C, max_iter=5000).fit(Xo, y)
    w = sorted(zip(kept, full.coef_[0]), key=lambda t: -abs(t[1]))
    print("  top weights", {k: round(float(c), 3) for k, c in w[:15]})
    if a.out:
        prob = 1 / (1 + np.exp(-tst))
        save_predictions(RESULTS_DIR, a.out, oof=1 / (1 + np.exp(-oof)), test=prob, train_ids=ids,
                         test_ids=te_ids)
        save_cv_result(RESULTS_DIR, a.out, [], full_auc, metric_name="auc",
                       params={"members": kept, "C": C})
        write_submission(SUBMISSIONS_DIR, f"{a.out}.csv", te_ids, TARGET, prob)
        print(f"wrote {a.out}  OOF {full_auc:.6f}")


if __name__ == "__main__":
    main()
