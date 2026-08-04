"""MLP leg — the strongest available diversity source for the blend.

    python src/train_mlp.py --arm s4 --seeds 42

**This is why the imputation work matters even if it is flat for LightGBM.**
GBDTs handle NaN natively, so imputing can only help them at the margin; an MLP
cannot take NaN at all. An imputed surface is a *prerequisite* for a neural leg,
and a neural leg is a genuinely different inductive bias from three GBDTs — the
one thing a blend of GBDTs cannot manufacture for itself.

So `--arm` should be an imputing arm (`s3`/`s4`). Whatever the imputer leaves
behind — the four never-impute columns and the categoricals — is filled here
with fold-train medians / an explicit missing level. That is lossless given the
mask indicators those arms already carry.

`fit_mlp_fold` fits Yeo-Johnson + StandardScaler on the training split only and
deletes its own GPU tensors; `torch.cuda.empty_cache()` is called after every
fold here, per the repo's GPU memory rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import RAW_CATEGORICAL_COLS
from legs import run_leg
from run_arm import ARMS

from data_science_stuff.kaggle.io import competition_dirs, load_params
from data_science_stuff.kaggle.models import fit_mlp_fold

_, RESULTS_DIR, _ = competition_dirs(__file__)

DEFAULT_PARAMS: dict[str, object] = {
    "n_layers": 3,
    "first_size": 512,
    "shrink": 0.5,
    "dropout": 0.15,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "activation": "gelu",
}


def to_matrix(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Dense float32 matrices: numeric block then one-hot block, no NaN."""
    x_tr = frames[0]
    num_cols = [c for c in x_tr.columns if c not in RAW_CATEGORICAL_COLS]
    medians = {c: float(x_tr[c].median()) for c in num_cols}

    numeric, categorical = [], []
    for frame in frames:
        block = frame[num_cols].copy()
        for col in num_cols:
            # Fold-train median; NaN survives only where the imputer declined.
            block[col] = block[col].fillna(medians[col]).replace([np.inf, -np.inf], np.nan)
            block[col] = block[col].fillna(medians[col])
        numeric.append(block.to_numpy(dtype=np.float32))
        # Categories were unioned across train/test in build_matrices, so the
        # dummy columns line up; missing becomes its own explicit column.
        dummies = pd.get_dummies(
            frame[RAW_CATEGORICAL_COLS].astype(object).fillna("__NA__").astype(str),
            columns=RAW_CATEGORICAL_COLS,
        )
        categorical.append(dummies)

    all_cols = sorted(set().union(*[set(d.columns) for d in categorical]))
    cat_arrays = [
        d.reindex(columns=all_cols, fill_value=0).to_numpy(dtype=np.float32)
        for d in categorical
    ]
    mats = [np.hstack([n, c]) for n, c in zip(numeric, cat_arrays)]
    n_num = numeric[0].shape[1]
    return mats, list(range(n_num)), list(range(n_num, mats[0].shape[1]))


def make_fit(seed: int, params: dict[str, object]):  # noqa: ANN201
    def fit_model(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        mats, num_idx, cat_idx = to_matrix((x_tr, x_va, x_te))
        val, test = fit_mlp_fold(
            mats[0],
            y_tr.astype(np.int64),
            mats[1],
            y_va.astype(np.int64),
            mats[2],
            num_idx,
            cat_idx,
            params,
            n_classes=2,
            epochs=60,
            batch_size=8192,
            patience=8,
            use_class_weights=False,
            score_fn=lambda yy, p: float(roc_auc_score(yy, p[:, 1])),
        )
        # GPU memory rule: fit_mlp_fold frees its own tensors, the loop clears
        # the allocator so multi-fold runs do not fragment into an OOM.
        del mats
        torch.cuda.empty_cache()
        return val[:, 1], test[:, 1]

    return fit_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s4", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    if ARMS[args.arm][1] is None:
        print(f"warning: arm {args.arm} does no imputation; the MLP will see "
              "median-filled columns for every null.")
    params = load_params(RESULTS_DIR, DEFAULT_PARAMS, "best_params_mlp.json")
    print(f"mlp on arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        run_leg("mlp", args.arm, seed, make_fit(seed, dict(params)))


if __name__ == "__main__":
    main()
