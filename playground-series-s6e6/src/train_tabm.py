"""TabM (pytabkit) 5-fold CV — the strong-diverse-base bet (PS S6E6).

The blend is GBDT-saturated and gives every non-GBDT model ~0 weight (all ≤0.955);
cdeotte's 0.9702 stack leans on strong diverse bases incl. TabM/TabICL/RealMLP at
0.968-0.9697. RealMLP underperformed here (0.9508, STAR recall ~0.92; oversampling
didn't fix it). TabM (Gorishniy 2024, parameter-efficient MLP ensembling) is a
stronger, different architecture — the test is whether it natively handles the
STAR/low-z-GALAXY overlap that broke RealMLP.

Same pytabkit interface as RealMLP_TD: feed raw frame + cat_col_names (internal PLR
numeric embeddings + cat embeddings, own val split for early stopping → leakage-
safe). NO class_weight knob, so the imbalance lever is val_metric_name=
"1-balanced_accuracy" only (clean first test; if STAR lags like RealMLP we'll know
it's an NN-on-these-features limit, not config). batch_size bumped off the default
256 (which starves the GPU) with mild sqrt-scaled lr. Per-fold checkpointed,
GPU-mem rule. Gate argmax vs 0.9654; watch STAR recall.

Run:  python src/train_tabm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)

N_FOLDS = 5
N_CLASSES = 3
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048      # default 256 starves the GPU on 462k rows
LR = 0.004             # TabM default 0.002, mild sqrt-scale for the 8x batch
RUN = "tabm"


def train_fold(X_tr, y_tr, X_val, X_test, cat_cols):
    model = TabM_D_Classifier(
        device=DEVICE, random_state=RANDOM_STATE,
        val_metric_name="1-balanced_accuracy",
        batch_size=BATCH_SIZE, lr=LR, verbosity=1,
    )
    model.fit(X_tr, y_tr, cat_col_names=cat_cols)
    assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return val_proba, test_proba


def main() -> None:
    print(f"Device: {DEVICE}  batch_size={BATCH_SIZE} lr={LR}")
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    X = train_pl.select(feature_cols).to_pandas()
    X_test = test_pl.select(feature_cols).to_pandas()
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    test_ids = test_pl.to_pandas()["id"].to_numpy()
    print(f"X {X.shape}  cats {cat_cols}  classes {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores: list[float] = []
    ckpt = RESULTS_DIR / "_tabm_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)

    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        vck, tck = ckpt / f"f{fold}_val.npy", ckpt / f"f{fold}_test.npy"
        if vck.exists() and tck.exists():
            val_pred, test_pred = np.load(vck), np.load(tck)
            print(f"  Fold {fold}: loaded checkpoint")
        else:
            val_pred, test_pred = train_fold(X.iloc[tri], y[tri], X.iloc[vai], X_test, cat_cols)
            np.save(vck, val_pred); np.save(tck, test_pred)
        oof[vai] = val_pred
        test_proba += test_pred / N_FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_pred, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [RealMLP was 0.9508; gate vs lgbm 0.9654]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  [watch STAR vs RealMLP 0.92]")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels)
    for f in ckpt.glob("f*.npy"):
        f.unlink()
    ckpt.rmdir()
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
