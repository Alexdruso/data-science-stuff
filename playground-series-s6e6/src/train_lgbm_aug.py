"""LGBM with SDSS17 original-data augmentation — PS S6E6.

The model/blend plateaued at 0.9657 CV while the LB top is ~0.97. A gap that
size on a Playground Series competition is a structural lever, not tuning: the
synthetic data was generated from the real SDSS17 dataset, so adding the real
rows to TRAINING lets the model learn the true manifold the synthetic test only
approximates. (Lookup was ruled out — test rows are marginal-resampled, not
copies; see memory / the diagnostics.)

CV design (leakage-safe + comparable):
  - StratifiedKFold(rs=42) over PLAYGROUND train only → OOF on playground rows,
    directly comparable to the 0.9654 argmax incumbent.
  - Each fold trains on (playground train-fold ⋃ ALL original rows); validates on
    the pure playground val-fold. Original never enters a val fold.
  - CV magnitude inflates vs LB (val rows were seeded from originals now in
    train) — confirm the real gain with an LB submission, don't trust CV alone.

Original prep: drop SDSS id cols + the u=g=z=-9999 corruption row; null the two
synthetic categoricals (spectral_type, galaxy_population) that the generator
added and the original lacks — LGBM routes on missingness.

Run:  python src/train_lgbm_aug.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from baseline import LGBM_PARAMS, load_params
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 5
CORE_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def load_original() -> pl.DataFrame:
    """SDSS17 original, aligned to the playground schema (2 synthetic cats null)."""
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.filter(pl.col("u") > -1000)  # drop the u=g=z=-9999 corruption row
    orig = orig.select([*CORE_FEATURES, "class"]).with_columns([
        pl.lit(None, dtype=pl.String).alias("spectral_type"),
        pl.lit(None, dtype=pl.String).alias("galaxy_population"),
    ])
    # id only so build_features (sorts by id) runs; values are intra-frame only.
    return orig.with_row_index("id")


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    orig_raw = load_original()

    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    orig_pl = compute_group_features(train_raw, build_features(orig_raw))
    print(f"Playground train: {train_pl.shape}  test: {test_pl.shape}  "
          f"original (aug): {orig_pl.shape}")

    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS
    ]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]

    train_pd, test_pd, orig_pd = (df.to_pandas() for df in (train_pl, test_pl, orig_pl))
    # Lock category sets from playground train so train/orig/test encode identically.
    for col in cat_cols:
        cats = pd.CategoricalDtype(categories=train_pd[col].dropna().unique())
        train_pd[col] = train_pd[col].astype(cats)
        test_pd[col] = test_pd[col].astype(cats)
        orig_pd[col] = orig_pd[col].astype(cats)  # all-NaN → missing

    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    y_orig = le.transform(orig_pd[TARGET].to_numpy())
    print(f"Classes: {list(le.classes_)}  | original class counts: "
          f"{np.bincount(y_orig).tolist()}")

    X = train_pd[feature_cols]
    X_orig = orig_pd[feature_cols]
    X_test = test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()

    params = load_params()
    params["num_class"] = len(le.classes_)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), len(le.classes_)))
    test_proba = np.zeros((len(X_test), len(le.classes_)))
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # train fold = playground-train-fold ⋃ ALL original; val = pure playground
        X_tr = pd.concat([X.iloc[train_idx], X_orig], ignore_index=True)
        y_tr = np.concatenate([y[train_idx], y_orig])
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        oof_proba[val_idx] = model.predict_proba(X_val)
        test_proba += model.predict_proba(X_test) / N_FOLDS
        score = float(balanced_accuracy_score(y_val, np.argmax(oof_proba[val_idx], axis=1)))
        fold_scores.append(score)
        print(f"  Fold {fold} balanced_acc: {score:.4f}")

    oof_argmax = float(balanced_accuracy_score(y, np.argmax(oof_proba, axis=1)))
    rec = recall_score(y, np.argmax(oof_proba, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {oof_argmax:.4f}  "
          f"[incumbent lgbm 0.9654]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")

    tw, best = optimize_thresholds(oof_proba, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    run = "lgbm_aug"
    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{run}.json")
    save_cv_result(RESULTS_DIR, run, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{run}.npy", oof_proba)
    np.save(RESULTS_DIR / f"test_{run}.npy", test_proba)

    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / f"{run}.csv"
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(out, index=False)
    print(f"Submission saved → {out}")


if __name__ == "__main__":
    main()
