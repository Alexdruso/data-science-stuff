"""TabPFN-3 feasibility + pricing + fair-comparison probe (Day-11, user-requested).

Runs in the ISOLATED tabpfn venv (/tmp/tabpfn_venv) — SELF-CONTAINED, no repo
imports (that venv has tabpfn 8.1.0 + sklearn 1.9, not our package). TabPFN-3 is
license-gated: set TABPFN_TOKEN first (https://ux.priorlabs.ai → accept license →
API key).

Answers, mirroring the Day-9 TabPFN-v2 probe but on the GENERAL population and the
u-surface (the deployed u-family core):
  1. Does TabPFN-3 run on the 6 GB RTX 2060, and how fast (sec/1k rows)?
  2. Is its solo balanced accuracy on a fair held-out eval chunk even in the same
     league as the u-core (oof_lrstack_u3_r) on the SAME rows?
  3. Is it decorrelated (error-overlap) — i.e. worth a full OOF build as a leg?

Row-order invariant: train is sorted by id to align with the saved core OOF array.
Context = stratified subsample (S6E7_CTX, default 30k); eval = disjoint stratified
30k held-out (labels known). balance_probabilities=True (prior-correct for BA).

Run: TABPFN_TOKEN=... S6E7_CTX=30000 /tmp/tabpfn_venv/bin/python src/probe_tabpfn3.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RES = HERE.parent / "results"
CLASSES = ["at-risk", "fit", "unhealthy"]
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count",
       "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
CTX = int(os.environ.get("S6E7_CTX", "30000"))
N_EVAL = 30_000
CORE_KEY = "lrstack_u3_r"


def encode(df: pd.DataFrame, cat_maps: dict) -> np.ndarray:
    cols = []
    for c in NUM:
        cols.append(df[c].to_numpy(dtype=np.float64))
    for c in CAT:
        codes = df[c].map(cat_maps[c]).to_numpy(dtype=np.float64)
        cols.append(codes)  # NaN stays NaN (unseen/missing)
    return np.column_stack(cols)


def weighted_argmax(proba: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (proba * w).argmax(axis=1)


def main() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("set TABPFN_TOKEN first (license-gated).")
    import torch
    from tabpfn import TabPFNClassifier

    train = pd.read_csv(DATA / "train.csv").sort_values("id").reset_index(drop=True)
    y = train["health_condition"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    cat_maps = {
        c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique()))}
        for c in CAT
    }
    X = encode(train, cat_maps)
    cat_idx = list(range(len(NUM), len(NUM) + len(CAT)))

    # stratified disjoint context / eval split (row indices into the id-sorted frame)
    all_idx = np.arange(len(y))
    ctx_idx, rest_idx = train_test_split(
        all_idx, train_size=CTX, stratify=y, random_state=0
    )
    eval_idx, _ = train_test_split(
        rest_idx, train_size=N_EVAL, stratify=y[rest_idx], random_state=0
    )
    print(f"context {len(ctx_idx):,} | eval {len(eval_idx):,} rows  (CTX={CTX})")

    clf = TabPFNClassifier(
        device="cuda", n_estimators=2, balance_probabilities=True,
        categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
    )
    t0 = time.time()
    clf.fit(X[ctx_idx], y[ctx_idx])
    t_fit = time.time() - t0
    t0 = time.time()
    proba = np.zeros((len(eval_idx), 3))
    CH = 4096
    for i in range(0, len(eval_idx), CH):
        proba[i : i + CH] = clf.predict_proba(X[eval_idx[i : i + CH]])
    t_pred = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    yev = y[eval_idx]
    tab_argmax = bacc(yev, proba.argmax(1))
    # quick per-class decision-weight search (coarse grid, in-sample UB on eval)
    best_w, best_s = np.ones(3), -1.0
    grid = np.linspace(0.3, 1.7, 15)
    for wa in grid:
        for wu in grid:
            w = np.array([wa, 1.0, wu])
            s = bacc(yev, weighted_argmax(proba, w))
            if s > best_s:
                best_s, best_w = s, w
    tab_dw = best_s

    # core on the SAME eval rows
    core_oof = np.load(RES / f"oof_{CORE_KEY}.npy")[eval_idx]
    dwj = json.load(open(RES / f"decision_weights_{CORE_KEY}.json"))
    cw = (np.array(list(dwj["decision"].values())) if "decision" in dwj
          else np.array([dwj[c] for c in CLASSES]))
    core_pred = weighted_argmax(core_oof, cw)
    core_dw = bacc(yev, core_pred)

    tab_pred = weighted_argmax(proba, best_w)
    cw_wrong, tw_wrong = core_pred != yev, tab_pred != yev
    overlap = (cw_wrong & tw_wrong).sum() / max((cw_wrong | tw_wrong).sum(), 1)

    lines = [
        f"TabPFN-3 probe  (context {len(ctx_idx):,}, eval {len(eval_idx):,})",
        f"  feasibility: fit {t_fit:.1f}s, predict {t_pred:.1f}s "
        f"({1000 * t_pred / len(eval_idx):.2f}s/1k), peak GPU {peak:.2f} GiB",
        f"  TabPFN-3 solo  argmax {tab_argmax:.4f}   +dec-weights(UB) {tab_dw:.4f}",
        f"  u-core ({CORE_KEY}) same rows  dec-weighted {core_dw:.4f}",
        f"  delta vs core (dec-weighted UB): {tab_dw - core_dw:+.4f}",
        f"  error-overlap with core: {overlap:.1%}  "
        f"(prior NN legs ~92%; <~85% = decorrelated)",
        "",
        "VERDICT: solo in-league (delta >= -0.002) AND decorrelated (<85% overlap)"
        " => worth a full 5-fold OOF build; else priced dead like TabPFN-v2 (Day-9).",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (RES / "probe_tabpfn3.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
