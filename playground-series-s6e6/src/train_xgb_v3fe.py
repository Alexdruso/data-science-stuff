"""New diverse base: XGBoost on the CAT_v3 feature set (decorrelated from xgb_deotte/catboost_v3).

The v3 FE (catboost_v3_features) is a rich cat-heavy set that helped CatBoost via ordered target
statistics. Giving the SAME FE to XGBoost — a different family — with fold-safe multiclass target
encoding of the 153 categoricals yields a strong base that's decorrelated from xgb_deotte (different
FE) and catboost_v3 (different family + different cat handling). Original SDSS17 rows appended per
fold at weight 0.06 (same augmentation as catboost_v3). id-sorted for stack alignment.

GATE: OOF argmax ~0.967+ AND a positive stack delta over the current 8-base 0.97006.

Run:  python src/train_xgb_v3fe.py [--smoke]
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from catboost_v3_features import CLASSES, build_features_cat_v3
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED = 42
N_SPLITS = 5
ORIGINAL_WEIGHT = 0.06
TE_SMOOTH = 20.0
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def balanced_error_metric(y_true, y_pred, sample_weight=None):
    yp = y_pred if y_pred.ndim == 2 else y_pred.reshape(-1, len(CLASSES))
    return 1.0 - balanced_accuracy_score(np.asarray(y_true).astype(int), np.argmax(yp, axis=1))


def make_xgb_params(seed, n_est, es):
    return {
        "objective": "multi:softprob", "num_class": len(CLASSES),
        "eval_metric": balanced_error_metric, "tree_method": "hist", "device": "cuda",
        "learning_rate": 0.012, "n_estimators": n_est, "early_stopping_rounds": es,
        "max_depth": 0, "max_leaves": 72, "grow_policy": "lossguide", "max_bin": 512,
        "min_child_weight": 10, "gamma": 0.20, "reg_alpha": 0.30, "reg_lambda": 4.0,
        "subsample": 0.82, "colsample_bytree": 0.74, "colsample_bylevel": 0.86,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }


def inv_freq_w(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    wpc = np.float32(len(y)) / (np.float32(len(CLASSES)) * np.maximum(counts, 1.0))
    return wpc[y].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_est = 400 if args.smoke else 7000
    es = 60 if args.smoke else 180

    tr = pl.read_csv(DATA_DIR / "train.csv").sort("id").to_pandas()
    te = pl.read_csv(DATA_DIR / "test.csv").sort("id").to_pandas()
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.with_columns(pl.col("class").str.to_uppercase()).filter(
        pl.col("class").is_in(CLASSES)).to_pandas()
    y = tr["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    y_orig = orig["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    test_ids = te["id"].to_numpy()

    print(f"building v3 features...", flush=True)
    X, X_test, X_orig, cat_cols = build_features_cat_v3(tr, te, orig)
    num_cols = [c for c in X.columns if c not in cat_cols]
    Xc, Xc_test, Xc_orig = X[cat_cols].astype("category"), X_test[cat_cols].astype("category"), X_orig[cat_cols].astype("category")
    Xn = X[num_cols].to_numpy(np.float32)
    Xn_test = X_test[num_cols].to_numpy(np.float32)
    Xn_orig = X_orig[num_cols].to_numpy(np.float32)
    print(f"X={X.shape}, {len(num_cols)} numeric + {len(cat_cols)} cats -> TE multiclass", flush=True)

    oof = np.zeros((len(X), 3), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        print(f"===== fold {fold}/{len(folds)} =====", flush=True)
        enc = TargetEncoder(target_type="multiclass", smooth=TE_SMOOTH, cv=5, random_state=SEED)
        te_tr = enc.fit_transform(Xc.iloc[tri], y[tri]).astype(np.float32)
        te_va = enc.transform(Xc.iloc[vai]).astype(np.float32)
        te_te = enc.transform(Xc_test).astype(np.float32)
        te_or = enc.transform(Xc_orig).astype(np.float32)

        Xtr = np.hstack([Xn[tri], te_tr])
        Xva = np.hstack([Xn[vai], te_va])
        Xte = np.hstack([Xn_test, te_te])
        Xor = np.hstack([Xn_orig, te_or])

        X_fit = np.vstack([Xtr, Xor])
        y_fit = np.concatenate([y[tri], y_orig])
        w = inv_freq_w(y_fit)
        w[len(tri):] *= np.float32(ORIGINAL_WEIGHT)

        model = XGBClassifier(**make_xgb_params(SEED + fold, n_est, es))
        model.fit(X_fit, y_fit, sample_weight=w, eval_set=[(Xva, y[vai])], verbose=False)
        oof[vai] = model.predict_proba(Xva).astype("float32")
        if not args.smoke:
            test_pred += model.predict_proba(Xte).astype("float32") / N_SPLITS
        sc = balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))
        print(f"fold {fold} bal-acc {sc:.6f}  best_iter {model.best_iteration}", flush=True)
        del model, enc, X_fit, Xtr, Xva, Xte, Xor, te_tr, te_va, te_te, te_or
        gc.collect()

    if args.smoke:
        print("smoke OK")
        return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"\nxgb_v3fe OOF argmax {argmax:.5f}  thresh {best:.5f}  [xgb_deotte 0.96699 / catboost_v3 0.96891]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    np.save(RESULTS_DIR / "oof_xgb_v3fe.npy", oof)
    np.save(RESULTS_DIR / "test_xgb_v3fe.npy", test_pred)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / "threshold_weights_xgb_v3fe.json")
    save_cv_result(RESULTS_DIR, "xgb_v3fe", [], best, metric_name="balanced_acc")
    labels = [CLASSES[i] for i in np.argmax(test_pred * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, "xgb_v3fe.csv", test_ids, "class", labels)
    print("Saved → xgb_v3fe")


if __name__ == "__main__":
    main()
