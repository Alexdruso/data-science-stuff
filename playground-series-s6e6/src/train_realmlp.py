"""RealMLP (pytabkit) 5-fold CV training for PS S6E6 — Predicting Stellar Class.

The non-GBDT diversity bet, take 2. The hand-rolled MLP (train_mlp.py) sat ~0.01
below LGBM because a vanilla net fed raw scaled scalars can't match a GBDT's
per-feature non-linear binning. RealMLP (Holzmüller et al. 2024, "Better by
Default") closes that gap with numerical (PLR) embeddings, categorical
embeddings, learnable per-feature scaling, robust scaling + smooth clipping — all
internal to the model. So unlike train_mlp.py this script does NO manual
preprocessing: we hand RealMLP the raw feature frame + cat_col_names and let it
encode.

Leakage (CLAUDE.md): RealMLP is fit on the training fold ONLY. It carves its own
internal validation split for early stopping, so the OOF fold (val_idx) is never
seen during training — the OOF score stays honest. Row order follows
build_features() (sorted by id) so oof/test .npy arrays align with every other
model's.

Class imbalance: STAR (14%) is the bottleneck class. RealMLP_TD has no
class_weight knob, so we early-stop on val_metric_name="balanced_accuracy"
(targets the competition metric directly, the role class_weight="balanced" plays
for the GBDTs) and rely on the shared post-hoc threshold optimisation + the blend.
If STAR recall lags in the diversity report, revisit with sample weights.

GPU-memory rule (CLAUDE.md): each fold dels its model and the outer loop calls
torch.cuda.empty_cache().

Run with:  python src/train_realmlp.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_FOLDS = 5
N_CLASSES = 3
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RealMLP_TD's default batch_size=256 starves the GPU on 462k rows (~10% util,
# ~5h, peak <0.3 GB) — it's launch-overhead bound on ~460k micro-batches. We
# raise batch_size 16× and sqrt-scale the LR (0.04→~0.15) to match. Measured:
# full run ~25-40 min, peak ~1.3 GB of 6 GB at bs=8192 (bs=4096 ≈ 0.8 GB), so
# memory (our tightest constraint) is comfortable. Trade-off: a deliberate
# deviation from the "tuned defaults" — validated on the real 5-fold OOF, not
# assumed. Schedule shape + PLR embeddings are unchanged.
BATCH_SIZE = 4096
LR = 0.15


def train_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit RealMLP on the train split; return (val_proba, test_proba) as (n × 3).

    RealMLP makes its own internal validation split from (X_tr, y_tr) for early
    stopping — X_val is only ever passed to predict_proba, never to fit.
    """
    model = RealMLP_TD_Classifier(
        device=DEVICE,
        random_state=RANDOM_STATE,
        val_metric_name="1-balanced_accuracy",
        batch_size=BATCH_SIZE,
        lr=LR,
        verbosity=1,
    )
    model.fit(X_tr, y_tr, cat_col_names=cat_cols)
    # predict_proba columns MUST be [0,1,2] == [GALAXY,QSO,STAR] to align with
    # every other model's .npy arrays + the threshold weights. A wrong order
    # fails silently into a garbage ensemble/submission, so assert it.
    assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return val_proba, test_proba


def main() -> None:
    print(f"Device: {DEVICE}")

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    print(f"Train: {train_pl.shape}   Test: {test_pl.shape}")

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    X = train_pl.select(feature_cols).to_pandas()
    X_test = test_pl.select(feature_cols).to_pandas()
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    test_ids = test_pl.to_pandas()["id"].to_numpy()
    print(f"Features: {len(feature_cols)}  categoricals: {cat_cols}  "
          f"classes: {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores: list[float] = []

    # Per-fold checkpointing: a ~1h/fold run must survive a mid-run shutdown.
    # Folds are deterministic (StratifiedKFold random_state=42), so on restart we
    # reload any fold already on disk and skip retraining it. Temp files are
    # cleaned up once the final assembled arrays are saved.
    ckpt_dir = RESULTS_DIR / "_realmlp_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        val_ckpt = ckpt_dir / f"fold{fold}_val.npy"
        test_ckpt = ckpt_dir / f"fold{fold}_test.npy"
        if val_ckpt.exists() and test_ckpt.exists():
            val_pred, test_pred = np.load(val_ckpt), np.load(test_ckpt)
            print(f"  Fold {fold}: loaded checkpoint (skipped training)")
        else:
            val_pred, test_pred = train_fold(
                X.iloc[train_idx], y[train_idx], X.iloc[val_idx], X_test, cat_cols,
            )
            np.save(val_ckpt, val_pred)
            np.save(test_ckpt, test_pred)
        oof_proba[val_idx] = val_pred
        test_proba += test_pred / N_FOLDS
        score = float(balanced_accuracy_score(y[val_idx], np.argmax(val_pred, axis=1)))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    oof_score = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))
    print(f"\nOOF balanced_acc (argmax): {oof_score:.4f}")

    threshold_weights, best_score = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best_score:.4f}")
    save_threshold_weights(
        threshold_weights, le.classes_.tolist(),
        RESULTS_DIR / "threshold_weights_realmlp.json",
    )
    save_cv_result(
        RESULTS_DIR, "realmlp_v1", fold_scores, best_score, metric_name="balanced_acc"
    )

    np.save(RESULTS_DIR / "oof_realmlp.npy", oof_proba)
    np.save(RESULTS_DIR / "test_realmlp.npy", test_proba)
    print(f"OOF/test arrays saved → {RESULTS_DIR}")

    # All folds assembled & saved — drop the per-fold checkpoints.
    for f in ckpt_dir.glob("fold*_*.npy"):
        f.unlink()
    ckpt_dir.rmdir()

    test_pred_labels = le.inverse_transform(
        np.argmax(test_proba * threshold_weights, axis=1)
    )
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "realmlp_v1.csv"
    pd.DataFrame({"id": test_ids, TARGET: test_pred_labels}).to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
