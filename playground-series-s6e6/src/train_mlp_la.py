"""MLP with LOGIT-ADJUSTED loss (balanced softmax) — the NN STAR fix (PS S6E6).

cdeotte's RealMLP reaches CV 0.9688 (strong + diverse) where ours/TabM/TabICL all
stall at ~0.95 with STAR recall ~0.91. The lever is NOT class weights (he turned
them OFF, as we found), NOT oversampling (OFF), but `loss_prior_power=1.075` —
balanced-softmax / logit adjustment: during training add tau·log(prior_c) to the
class logits before cross-entropy; predict with RAW logits. This shifts the
decision boundary toward rare STAR without distorting inference probabilities.

The mechanism now lives in the shared package: fit_mlp_fold(logit_adjust=...,
use_class_weights=False) with logit_adjustment() computing the tau-scaled,
geomean-normalized prior offsets from the FOLD's training labels.

Run:  python src/train_mlp_la.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights
from train_mlp import (
    BATCH_SIZE, DEFAULT_PARAMS, DEVICE, EPOCHS, N_CLASSES, N_FOLDS, PATIENCE,
    prepare_arrays,
)

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.models.losses import logit_adjustment
from data_science_stuff.kaggle.models.mlp import fit_mlp_fold

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
TAU = 1.075   # cdeotte's loss_prior_power
RUN = "mlp_la"


def main() -> None:
    print(f"Device: {DEVICE}  TAU={TAU}")
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    X, X_test, num_idx, cat_idx = prepare_arrays(train_pl, test_pl, cat_cols, feature_cols)
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    test_ids = test_pl.to_pandas()["id"].to_numpy()
    print(f"X {X.shape}  classes {list(le.classes_)}")

    def fit_fold(x_tr, y_tr, x_va, y_va, x_te, _fold):
        # Priors are fold statistics: compute the adjustment from THIS fold's
        # training labels. No class weights, no oversampling (both confirmed flat).
        adj = logit_adjustment(np.bincount(y_tr, minlength=N_CLASSES), tau=TAU)
        return fit_mlp_fold(
            x_tr, y_tr, x_va, y_va, x_te, num_idx, cat_idx, DEFAULT_PARAMS,
            n_classes=N_CLASSES, device=DEVICE,
            epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
            use_class_weights=False, logit_adjust=adj,
        )

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(balanced_accuracy_score(y_va, np.argmax(val_pred, axis=1)))

    def free_gpu(_fold: int) -> None:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    oof, test_proba, fold_scores = run_cv(
        X, y, X_test, fit_fold,
        n_splits=N_FOLDS, n_outputs=N_CLASSES,
        score_fn=fold_score, after_fold=free_gpu,
    )
    for fold, score in enumerate(fold_scores, 1):
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [plain MLP 0.9547; deotte-realmlp 0.9688]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  [watch STAR vs plain-MLP 0.958]")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
