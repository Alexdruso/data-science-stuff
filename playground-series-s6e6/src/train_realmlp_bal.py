"""RealMLP with per-fold minority OVERSAMPLING — the strong-diverse-base unlock.

Our RealMLP hit 0.9508 with STAR recall 0.91-0.93 while GALAXY recall was fine —
a broken minority-class config, not a weak family. RealMLP_TD has no class_weight/
sample_weight knob, so the fix is resampling: per fold, oversample STAR (and QSO)
in the TRAIN split to balance GALAXY, then fit. If STAR recall climbs to ~0.96 the
model jumps toward 0.965 — finally giving the blend a STRONG + DIVERSE non-GBDT
base (every non-GBDT we have is ≤0.955 and gets ~0 blend weight; Deotte's 0.9702
stack leans on a RealMLP at 0.9697).

Leakage-safe: oversample only the train-fold; the pure val-fold is untouched and is
what OOF scores. Reuses train_realmlp.train_fold (bs=4096/lr=0.15, GPU-mem rule,
classes_ assertion) so only the resampling differs.

Run:  python src/train_realmlp_bal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights
from train_realmlp import N_CLASSES, RANDOM_STATE, train_fold

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5
RUN = "realmlp_bal"


def balanced_indices(y_tr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices (into y_tr) that oversample every class with replacement to the
    majority count → ~balanced classes."""
    counts = np.bincount(y_tr, minlength=N_CLASSES)
    target = int(counts.max())
    parts = []
    for c in range(N_CLASSES):
        c_idx = np.where(y_tr == c)[0]
        if len(c_idx) < target:
            extra = rng.choice(c_idx, target - len(c_idx), replace=True)
            parts.append(np.concatenate([c_idx, extra]))
        else:
            parts.append(c_idx)
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def main() -> None:
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
    print(f"X {X.shape}  classes {list(le.classes_)}  base counts {np.bincount(y).tolist()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores: list[float] = []
    rng = np.random.default_rng(RANDOM_STATE)
    ckpt = RESULTS_DIR / "_realmlp_bal_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)

    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        vck, tck = ckpt / f"f{fold}_val.npy", ckpt / f"f{fold}_test.npy"
        if vck.exists() and tck.exists():
            val_pred, test_pred = np.load(vck), np.load(tck)
            print(f"  Fold {fold}: loaded checkpoint")
        else:
            bal = balanced_indices(y[tri], rng)
            X_tr_bal = X.iloc[tri].iloc[bal].reset_index(drop=True)
            y_tr_bal = y[tri][bal]
            val_pred, test_pred = train_fold(X_tr_bal, y_tr_bal, X.iloc[vai], X_test, cat_cols)
            np.save(vck, val_pred); np.save(tck, test_pred)
        oof[vai] = val_pred
        test_proba += test_pred / N_FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_pred, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [plain RealMLP 0.9508, STAR 0.93]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")
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
