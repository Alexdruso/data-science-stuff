"""TabPFN-3 full 5-fold OOF leg (Day-11) — the probe passed the gate.

The feasibility/pricing probe (`probe_tabpfn3.py`) cleared both bars: solo
dec-weighted 0.9494 vs u-core 0.9506 (−0.0012, in-league) AND error-overlap 80.2%
(< 85%, the most decorrelated base of the campaign). ⇒ build a proper leg and
seat-test it in the u-stack.

Runs in the ISOLATED tabpfn venv (self-contained, no repo imports). Produces
leak-free OOF via the SAME CV as every other leg — StratifiedKFold(5, shuffle,
random_state=42) on the label, id-sorted rows — so the arrays stack with the
u-family. Per fold, TabPFN-3's context = a stratified subsample of the fold's
TRAIN part (S6E7_CTX, default 50k; the val part is never in context → leak-free);
test = fold-averaged. Arrays saved id-sorted for the row-order invariant.

Gate afterwards in the MAIN venv: decision weights + signature + 12-cell seat test.

Run: TABPFN_TOKEN=... S6E7_CTX=50000 /tmp/tabpfn_venv/bin/python src/train_tabpfn3.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
CLASSES = ["at-risk", "fit", "unhealthy"]
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count",
       "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
CTX = int(os.environ.get("S6E7_CTX", "50000"))
NEST = int(os.environ.get("S6E7_NEST", "2"))
KEY = "tabpfn3_u_s42"


def encode(df: pd.DataFrame, cat_maps: dict) -> np.ndarray:
    cols = [df[c].to_numpy(dtype=np.float64) for c in NUM]
    for c in CAT:
        cols.append(df[c].map(cat_maps[c]).to_numpy(dtype=np.float64))
    return np.column_stack(cols)


def main() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("set TABPFN_TOKEN first (license-gated).")
    from tabpfn import TabPFNClassifier

    train = pd.read_csv(DATA / "train.csv").sort_values("id").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").sort_values("id").reset_index(drop=True)
    y = train["health_condition"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    cat_maps = {
        c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique()))}
        for c in CAT
    }
    X = encode(train, cat_maps)
    Xte = encode(test, cat_maps)
    cat_idx = list(range(len(NUM), len(NUM) + len(CAT)))
    print(f"context/fold={CTX:,}  n_estimators={NEST}  train {len(y):,}  test {len(Xte):,}")

    oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)

    def subctx(idx):  # noqa: ANN001, ANN202
        if len(idx) <= CTX:
            return idx
        return np.concatenate([
            rng.permutation(idx[y[idx] == c])[: max(1, int(CTX * (y[idx] == c).mean()))]
            for c in range(3)
        ])

    def predict_chunked(clf, M):  # noqa: ANN001, ANN202
        out = np.zeros((len(M), 3))
        for i in range(0, len(M), 4096):
            out[i : i + 4096] = clf.predict_proba(M[i : i + 4096])
        return out

    def new_clf():  # noqa: ANN202
        return TabPFNClassifier(
            device="cuda", n_estimators=NEST, balance_probabilities=True,
            categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
        )

    # 5-fold OOF (leak-free: context = train-part subsample, val never in context)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        ctx = subctx(tr)
        clf = new_clf()
        t0 = time.time()
        clf.fit(X[ctx], y[ctx])
        oof[va] = predict_chunked(clf, X[va])
        print(f"  fold {fold}: ctx {len(ctx):,}  {time.time() - t0:.0f}s", flush=True)
        del clf
    np.save(RES / f"oof_{KEY}.npy", oof)

    # test predicted ONCE from a full-train context (test-pred does not need CV avg)
    ctx_all = subctx(np.arange(len(y)))
    clf = new_clf()
    clf.fit(X[ctx_all], y[ctx_all])
    test_p = predict_chunked(clf, Xte)
    del clf
    np.save(RES / f"test_{KEY}.npy", test_p)
    from sklearn.metrics import balanced_accuracy_score as bacc
    print(f"saved oof_{KEY}.npy / test_{KEY}.npy  | argmax OOF bacc {bacc(y, oof.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
