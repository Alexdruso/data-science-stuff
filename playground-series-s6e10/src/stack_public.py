"""LR-on-logits stack of our legs, optionally with public OOF/test pairs (PR-21).

    python src/stack_public.py --ours LEG ... [--public DIR ...] [--C 0.05] [--write NAME]

Public pairs (data/public_oof/<dataset>/oof_<m>.npy + test_<m>.npy) were produced
on StratifiedKFold(5, shuffle=True, random_state=42) — the same folds as ours —
in train.csv order, which is id order = build_features() order (checked
2026-10-08). The stacker is fit on 4 folds of the OOF matrix and predicts the
5th (same folds), so its OOF is nested; test = mean of the 5 fold stackers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def _logit(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64).ravel()
    if a.min() < 0 or a.max() > 1:
        return a  # already logits
    q = np.clip(a, 1e-7, 1 - 1e-7)
    return np.log(q / (1 - q))


def load_pairs(ours: list[str], public: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, oofs, tests = [], [], []
    for leg in ours:
        names.append(leg)
        oofs.append(_logit(np.load(RESULTS_DIR / f"oof_{leg}.npy")))
        tests.append(_logit(np.load(RESULTS_DIR / f"test_{leg}.npy")))
    for d in public:
        for f in sorted(Path(d).glob("oof_*.npy")):
            t = f.with_name(f.name.replace("oof_", "test_", 1))
            if not t.exists():
                continue
            o, tt = np.load(f), np.load(t)
            if o.ndim != 1 or len(o) != len(oofs[0] if oofs else o) or np.isnan(o).any() or np.isnan(tt).any():
                continue
            names.append(f"{Path(d).name[:12]}/{f.stem[4:]}")
            oofs.append(_logit(o))
            tests.append(_logit(tt))
    return names, np.column_stack(oofs), np.column_stack(tests)


def stack(y: np.ndarray, O: np.ndarray, T: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof, test = np.zeros(len(y)), np.zeros(len(T))
    coefs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(O, y):
        sc = StandardScaler().fit(O[tr])
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(O[tr]), y[tr])
        oof[va] = lr.decision_function(sc.transform(O[va]))
        test += lr.decision_function(sc.transform(T)) / 5
        coefs.append(lr.coef_.ravel())
    return oof, test, np.mean(coefs, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", nargs="*", default=[])
    p.add_argument("--C", type=float, nargs="+", default=[0.05])
    p.add_argument("--write", default="")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()

    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    names, O, T = load_pairs(a.ours, a.public)
    print(f"{len(names)} members")
    best = None
    for C in a.C:
        oof, test, coef = stack(y, O, T, C)
        auc = float(roc_auc_score(y, oof))
        print(f"C={C}: stack OOF {auc:.6f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, oof, test, coef, C)
    assert best is not None
    auc, oof, test, coef, C = best
    order = np.argsort(-np.abs(coef))[: a.top]
    for i in order:
        print(f"   {coef[i]:+.3f}  {names[i]}  (solo {roc_auc_score(y, O[:, i]):.6f})")
    if a.write:
        np.save(RESULTS_DIR / f"oof_{a.write}.npy", oof)
        np.save(RESULTS_DIR / f"test_{a.write}.npy", test)
        test_ids = build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()
        out = write_submission(SUBMISSIONS_DIR, f"{a.write}.csv", test_ids, TARGET, 1 / (1 + np.exp(-test)))
        print(f"best C={C} OOF {auc:.6f} → {out}")


if __name__ == "__main__":
    main()
