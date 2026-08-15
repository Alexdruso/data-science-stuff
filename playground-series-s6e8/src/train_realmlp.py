"""RealMLP-TD leg — the last untested model *class* in this competition.

    python src/train_realmlp.py --arm s4 --seeds 42
    python src/train_realmlp.py --arm s4 --fold1        # one-fold smoke, saves nothing

**Why this exists, stated honestly.** `mlp_s4` scored 0.941983 and earned blend weight
0.0, and `bag(cat×3)` at 0.962516 — only 0.0018 below the null — *also* earned 0.0. So
the prior for any neural leg here is weight 0.0, and "it is decorrelated" is not an
argument (the MLP was the most decorrelated leg by far and was harmful at every weight).
The one fact that keeps the question open: 0.941983 sits barely above the **0.9277**
fitted-linear ceiling, which is anomalously bad for a net on 691k rows — that gap is
either the architecture or the substrate, and RealMLP-TD is the strongest available test
of the architecture half.

**Two package facts drive the wiring below, and both are traps:**

1. `NumericalPreprocessor` does **not** impute — its transforms are `median_center` +
   `robust_scale`, and `np.median` over a column containing NaN yields NaN, which then
   poisons that whole column. It is *less* forgiving than `fit_mlp_fold`, which at least
   fits Yeo-Johnson NaN-blind. Whatever the arm's imputer declines is filled here with
   **fold-train medians**, exactly as s6e7's `train_realmlp_td.py` does.
2. `fit()` does `X[cat_col_names].to_numpy().astype(np.int64)`, so categoricals must
   arrive as integer **codes**, not pandas `category` dtype. And `cat_dims` is derived
   from the observed max, after which codes are `np.clip(codes, 0, cat_max)` — so a
   pandas `-1` NaN code would be silently merged into real level 0. Codes are shifted
   by +1 here so missing gets a level of its own.

GPU memory rule: `RealMLP_TD_Classifier.fit` ends at `return self` and frees nothing, so
the model is deleted and the allocator cleared inside every fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_CATEGORICAL_COLS
from legs import N_FOLDS, run_leg
from run_arm import ARMS, assert_budget_constraint, build_matrices

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.models import REALMLP_TD_CONFIG, RealMLP_TD_Classifier

_, RESULTS_DIR, _ = competition_dirs(__file__)
CKPT_DIR = RESULTS_DIR / "realmlp_ckpt"

# Overrides on top of REALMLP_TD_CONFIG. Everything not named here stays at the
# s6e6/s6e7-proven default (n_ens=8, ema_decay=0.99775, flat_cos, ...).
CFG: dict[str, object] = {
    # `bacc` thresholds at argmax and `bll` optimises calibration; on a 71%-positive
    # binary target under ROC AUC both select the wrong epoch. Added to the package
    # for this competition (binary-only, validated at construction).
    "epoch_select": "auc",
    # The 1.075 balanced-softmax prior is an imbalance hack. For 2 classes it is a
    # monotone reparameterisation, so it cannot change the *ranking* at eval — but it
    # does reshape the training gradient, and there is no imbalance here worth paying
    # for at a 0.709 base rate.
    "loss_prior_power": 0.0,
    # s6e7's SPEED_CFG on ~620k rows: ~3 min/fold. Stock train_bs=256 is tuned for
    # far smaller tables and would roughly double the wall clock on 553k rows/fold.
    "train_bs": 512,
    "allow_amp": True,
    "fused_optimizer": True,
    "verbosity": 1,
}


def to_frames(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[str]]:
    """NaN-free frames with categoricals as non-negative integer codes.

    Medians come from the fold's *training* split only — these are fold-aware
    statistics and fitting them on the full matrix would leak (root CLAUDE.md).
    """
    x_tr = frames[0]
    cat_cols = [c for c in RAW_CATEGORICAL_COLS if c in x_tr.columns]
    num_cols = [c for c in x_tr.columns if c not in cat_cols]

    clean_tr = x_tr[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = clean_tr.median()
    # A column that is all-NaN on the training split has no median; 0.0 keeps it
    # finite and constant, which the preprocessor's IQR fallback then handles.
    medians = medians.fillna(0.0)

    out: list[pd.DataFrame] = []
    for frame in frames:
        block = frame[num_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
        for col in cat_cols:
            # `build_matrices` unioned the level sets across train and test, so codes
            # are comparable. +1 moves pandas' -1 for NaN up to 0, giving missing its
            # own level instead of letting fit()'s clip merge it into real level 0.
            block[col] = frame[col].cat.codes.to_numpy() + 1
        out.append(block)
    return out, cat_cols


def make_fit(seed: int, cfg: dict[str, object]):  # noqa: ANN201 - closure
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        (tr, va, te), cat_cols = to_frames((x_tr, x_va, x_te))

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        model = RealMLP_TD_Classifier(**{**cfg, "random_state": seed + fold})
        model.fit(
            tr,
            y_tr.astype(np.int64),
            va,
            y_va.astype(np.int64),
            cat_col_names=cat_cols,
            # Per fold: the default path is shared per-PID, so five folds in one
            # process would overwrite each other's best checkpoint.
            ckpt_path=CKPT_DIR / f"s{seed}_f{fold}.pth",
            # Passed only so cat_dims covers a level that appears solely in test.
            X_test=te,
        )
        val = np.asarray(model.best_val_probs_)[:, 1]
        test = model.predict_proba(te)[:, 1]

        del model, tr, va, te
        torch.cuda.empty_cache()
        return val, test

    return fit_model


def smoke_fold1(arm: str, seed: int, cfg: dict[str, object]) -> float:
    """Fit fold 1 only and report its AUC. Saves nothing; for wiring + substrate reads."""
    builder, transform, _ = ARMS[arm]
    x, y, x_test = build_matrices(builder)
    assert_budget_constraint(x)

    # Same splitter as `run_cv`, so this fold's AUC is comparable to fold 1 of a
    # full run at the same seed.
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(iter(splitter.split(np.zeros(len(x)), y)))
    x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    x_te = x_test
    if transform is not None:
        x_tr, x_va, x_te = transform(x_tr, y_tr, x_va, x_te)
        if x_tr.shape[1] == x.shape[1]:
            raise RuntimeError(f"arm {arm}: transform added no columns — silent no-op")

    val, _ = make_fit(seed, cfg)(x_tr, y_tr, x_va, y_va, x_te, 1)
    auc = float(roc_auc_score(y_va, val))
    print(
        f"  realmlp_{arm}_s{seed} FOLD 1 ONLY: features={x_tr.shape[1]:>3} "
        f"(surface {x.shape[1]})  AUC {auc:.5f}",
        flush=True,
    )
    return auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s4", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--fold1",
        action="store_true",
        help="fit fold 1 only and print its AUC; saves no arrays and touches no scores",
    )
    parser.add_argument("--verbose-epochs", action="store_true")
    args = parser.parse_args()

    cfg = {**REALMLP_TD_CONFIG, **CFG}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.verbose_epochs:
        cfg["verbosity"] = 2

    print(f"realmlp on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(
        f"epochs={cfg['epochs']} train_bs={cfg['train_bs']} n_ens={cfg['n_ens']} "
        f"epoch_select={cfg['epoch_select']}",
        flush=True,
    )
    for seed in (int(s) for s in args.seeds.split(",")):
        if args.fold1:
            smoke_fold1(args.arm, seed, cfg)
        else:
            # `cfg` is the checkpoint key's config component. This leg is the slowest
            # in the competition (~10 min/fold) and the 08-13 run died at fold 2, so
            # fold-level restore matters more here than anywhere else. Distinct from
            # the per-fold `ckpt_path` above, which is RealMLP's own within-fit best
            # -epoch snapshot and does not survive the fold ending.
            run_leg("realmlp", args.arm, seed, make_fit(seed, cfg), config=cfg)


if __name__ == "__main__":
    main()
