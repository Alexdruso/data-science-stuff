"""One-fold diagnostic for the CatBoost leg's unexplained -0.0042.

    python src/diag_catboost.py

`cat_s1` seed 42 landed at OOF AUC **0.95953**. CLAUDE.md item 9 pre-registered
"solo OOF within +/-0.0005 of 0.96373" for it, so this misses its own stated
prior by ~8x. The Day-6 session crashed before that could be looked at, and an
unexplained 8x deviation is not something to carry into a blend silently.

Two concrete, cheap hypotheses, both created by the 6GB-GPU recipe in
`train_catboost.py` rather than by CatBoost being weak:

**H1 - border_count.** CatBoost defaults to `border_count=254` on CPU but
**128 on GPU**. This dataset's entire signal sits in fine-grained continuous
columns (single-feature AUC: daily 0.889, weekend 0.880, social 0.858), and
LightGBM reaches 0.96373 with `max_bin=255`. Halving the split resolution on
exactly the columns that carry the signal is a plausible four-figure loss.

**H2 - no early stopping.** The recipe drops `eval_set` (it triggers a silent
CPU fallback), so the iteration count is a fixed 2000 rather than searched.
LightGBM early-stops at ~600-900 trees here. 2000 fixed could be off in either
direction.

The two are separable and the grid is 2x2-ish, so one fold answers it. Wall
time is reported alongside AUC because the whole reason the recipe exists is
that the alternatives are slow -- a config that wins by 0.002 and costs 40
minutes a fold is not a win at this stage of the campaign.

Reference point printed alongside: LightGBM `s1` on the same fold.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_CATEGORICAL_COLS
from run_arm import ARMS, LGBM_PARAMS, build_matrices

SEED = 42
N_FOLDS = 5

BASE: dict[str, object] = {
    "iterations": 2000,
    "learning_rate": 0.06,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "max_ctr_complexity": 1,
    "verbose": 0,
    "allow_writing_files": False,
    "random_seed": SEED,
}

#: (label, overrides, use_eval_set)
CONFIGS: list[tuple[str, dict[str, object], bool]] = [
    ("A  GPU border=128 (SHIPPED)", {"task_type": "GPU", "gpu_ram_part": 0.8}, False),
    (
        "B  GPU border=254",
        {"task_type": "GPU", "gpu_ram_part": 0.8, "border_count": 254},
        False,
    ),
    ("C  CPU border=254 + early stop", {"task_type": "CPU", "border_count": 254}, True),
    ("D  CPU border=254 fixed 2000", {"task_type": "CPU", "border_count": 254}, False),
]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in RAW_CATEGORICAL_COLS:
        out[col] = out[col].astype(object).fillna("__NA__").astype(str)
    return out


def main() -> None:
    # The reference fit is the expensive part (~17 min under load), and it is a
    # constant for a fixed fold — so a re-run that only needs the CatBoost side
    # can pass the known value instead of paying for it twice.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="",
        help="comma-separated config letters to run, e.g. 'B,D' (default: all)",
    )
    parser.add_argument(
        "--reference",
        type=float,
        default=None,
        help="skip the LightGBM reference fit and use this fold-1 AUC instead",
    )
    args = parser.parse_args()
    wanted = {c.strip().upper() for c in args.configs.split(",") if c.strip()}

    x, y, _ = build_matrices(ARMS["s1"][0])
    tr_idx, va_idx = next(
        iter(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x, y))
    )
    x_tr, x_va = prepare(x.iloc[tr_idx]), prepare(x.iloc[va_idx])
    y_tr, y_va = y[tr_idx], y[va_idx]
    print(f"fold 1 of {N_FOLDS}: train {x_tr.shape}, val {x_va.shape}\n", flush=True)

    # Same-fold LightGBM reference, so the gap is read against this fold rather
    # than against a 5-fold OOF number computed on different rows.
    if args.reference is not None:
        ref = args.reference
        print(f"  REFERENCE  lgbm s1        AUC {ref:.5f}  (supplied, not refit)\n", flush=True)
    else:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: PLC0415

        t0 = time.time()
        lgbm = LGBMClassifier(**{**LGBM_PARAMS, "random_state": SEED})
        lgbm.fit(
            x.iloc[tr_idx],
            y_tr,
            eval_set=[(x.iloc[va_idx], y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        ref = roc_auc_score(y_va, lgbm.predict_proba(x.iloc[va_idx])[:, 1])
        print(
            f"  REFERENCE  lgbm s1        AUC {ref:.5f}  "
            f"{time.time() - t0:6.1f}s  best_iter={lgbm.best_iteration_}\n",
            flush=True,
        )

    results: list[tuple[str, float, float]] = []
    for label, overrides, use_eval in CONFIGS:
        if wanted and label[0] not in wanted:
            continue
        params = {**BASE, **overrides}
        t0 = time.time()
        try:
            model = CatBoostClassifier(**params)
            if use_eval:
                model.fit(
                    x_tr,
                    y_tr,
                    cat_features=RAW_CATEGORICAL_COLS,
                    eval_set=(x_va, y_va),
                    early_stopping_rounds=100,
                )
            else:
                model.fit(x_tr, y_tr, cat_features=RAW_CATEGORICAL_COLS)
            auc = float(roc_auc_score(y_va, model.predict_proba(x_va)[:, 1]))
            best = getattr(model, "best_iteration_", None)
            elapsed = time.time() - t0
            results.append((label, auc, elapsed))
            print(
                f"  {label:32} AUC {auc:.5f}  {elapsed:6.1f}s  "
                f"vs lgbm {auc - ref:+.5f}  best_iter={best}",
                flush=True,
            )
            del model
        except Exception as exc:  # noqa: BLE001 - a config failing is a result
            print(f"  {label:32} FAILED after {time.time() - t0:.1f}s: {exc}", flush=True)

    if results:
        best_label, best_auc, _ = max(results, key=lambda r: r[1])
        shipped = next((a for lbl, a, _ in results if lbl.startswith("A")), None)
        print(f"\nbest config: {best_label}  AUC {best_auc:.5f}")
        if shipped is not None:
            print(f"gain over the shipped config: {best_auc - shipped:+.5f}")
            print(f"remaining gap to lgbm on this fold: {best_auc - ref:+.5f}")


if __name__ == "__main__":
    main()
