"""RealMLP-TD leg for PS S6E10 — a model that differs in kind from the GBDTs.

    python src/train_realmlp.py               # 5-fold, saves oof/test_realmlp_v1.npy
    python src/train_realmlp.py --epochs 24
    python src/train_realmlp.py --fold1 --tag e8 --override '{"epochs": 8}'

Wiring follows s6e8's `train_realmlp.py`, including its two package traps:

1. `NumericalPreprocessor` does not impute (a NaN median poisons the column), so
   numeric NaNs are filled with **fold-train medians** here.
2. `fit()` casts categoricals with `.astype(np.int64)` and clips to the observed
   max, so they must arrive as integer codes; +1 gives a pandas -1 (NaN) its own
   level instead of merging it into level 0.

`epoch_select="auc"` matters: on s6e8 RealMLP peaked mid-run and then overfit
monotonically. Note the selected epoch is chosen on the val fold it is scored on,
so solo OOF is slightly optimistic (see CLAUDE.md PR-2).

GPU memory rule: `fit` frees nothing, so the model is deleted and the allocator
cleared inside every fold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET
from matrices import build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.models import REALMLP_TD_CONFIG, RealMLP_TD_Classifier

_, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
CKPT_DIR = RESULTS_DIR / "realmlp_ckpt"

# Overrides on top of REALMLP_TD_CONFIG (s6e8-proven for binary AUC).
CFG: dict[str, object] = {
    "epoch_select": "auc",
    # The balanced-softmax prior is an imbalance hack; 44% positive needs none.
    "loss_prior_power": 0.0,
    "train_bs": 512,
    "allow_amp": True,
    "fused_optimizer": True,
    "epochs": 16,
    "verbosity": 1,
}


def to_frames(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], cat_cols: list[str]
) -> list[pd.DataFrame]:
    """NaN-free frames, categoricals as non-negative integer codes.

    Medians come from the fold's training split only (fold-aware statistics).
    """
    num_cols = [c for c in frames[0].columns if c not in cat_cols]
    medians = frames[0][num_cols].median().fillna(0.0)
    out: list[pd.DataFrame] = []
    for frame in frames:
        block = frame[num_cols].astype(np.float64).fillna(medians)
        for col in cat_cols:
            block[col] = frame[col].cat.codes.to_numpy() + 1
        out.append(block)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", default="{}", help="JSON dict of config overrides")
    parser.add_argument("--tag", default="", help="run-name suffix; required with --override")
    parser.add_argument(
        "--fold1", action="store_true",
        help="fit fold 1 only (same split as run_cv) and print its AUC; saves no arrays",
    )
    args = parser.parse_args()

    overrides = json.loads(args.override)
    if overrides and not args.tag:
        raise SystemExit("--override needs --tag, or the arrays overwrite realmlp_v1")
    cfg = {**REALMLP_TD_CONFIG, **CFG, **overrides}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    m = build_matrices()
    # is_original is constant 0 here; RealMLP's robust scaler has nothing to scale.
    X = m.X.drop(columns=["is_original"])
    X_test = m.X_test.drop(columns=["is_original"])
    print(
        f"RealMLP on {'cuda' if torch.cuda.is_available() else 'cpu'}: "
        f"train {X.shape} test {X_test.shape} epochs={cfg['epochs']} "
        f"n_ens={cfg['n_ens']} train_bs={cfg['train_bs']}",
        flush=True,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        tr, va, te = to_frames((x_tr, x_va, x_te), m.cat_cols)
        model = RealMLP_TD_Classifier(**{**cfg, "random_state": args.seed + fold})
        model.fit(
            tr,
            y_tr.astype(np.int64),
            va,
            y_va.astype(np.int64),
            cat_col_names=m.cat_cols,
            # Per fold, or the five folds overwrite each other's best checkpoint.
            ckpt_path=CKPT_DIR / f"{args.tag or 'v1'}_s{args.seed}_f{fold}.pth",
            X_test=te,
        )
        val = np.asarray(model.best_val_probs_)[:, 1]
        test = model.predict_proba(te)[:, 1]
        print(
            f"  fold {fold}: AUC {roc_auc_score(y_va, val):.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del model, tr, va, te
        torch.cuda.empty_cache()
        return val, test

    if args.fold1:
        # Same splitter and seed as run_cv, so this is fold 1 of a full run.
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, va_idx = next(iter(splitter.split(X, m.y)))
        val, _ = fit_fold(
            X.iloc[tr_idx], m.y[tr_idx], X.iloc[va_idx], m.y[va_idx], X_test.iloc[:1000], 1
        )
        print(f"FOLD1 tag={args.tag or 'v1'} seed={args.seed} {overrides} "
              f"AUC {roc_auc_score(m.y[va_idx], val):.5f}", flush=True)
        return

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(
        X, m.y, X_test, fit_fold, n_outputs=1, score_fn=fold_score,
        after_fold=lambda _f: torch.cuda.empty_cache(),
    )
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"\nfolds {[round(s, 5) for s in fold_scores]}  OOF AUC: {oof_auc:.5f}")

    run_name = f"realmlp_{args.tag or 'v1'}" + ("" if args.seed == 42 else f"_s{args.seed}")
    save_cv_result(RESULTS_DIR, run_name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{run_name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run_name}.npy", test)
    out = write_submission(SUBMISSIONS_DIR, f"{run_name}.csv", m.test_ids, TARGET, test)
    print(f"Arrays → {RESULTS_DIR}   Submission → {out}")


if __name__ == "__main__":
    main()
