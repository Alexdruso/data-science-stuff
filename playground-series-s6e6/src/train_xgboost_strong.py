"""XGBoost in the STRONG-BOOSTER regime, mirroring cdeotte's XGB v1 (CV 0.9669) — S6E6.

Our XGB is 0.9645 and tuning it was "a wash" — because tune_xgboost.py (like tune_lgbm)
floored lr and the strong regime was never searched. cdeotte's XGB: lr=0.012, n_est=7000,
es≈180, grow_policy=lossguide max_leaves=72, balanced sample_weight, custom eval=1−balanced_acc.
We mirror the booster regime + the custom balanced-acc early-stop (default was mlogloss — a
proper loss, but not the target metric). Same base features (his +0.0024 FE was flat for us).

GATE: standalone argmax clears ~0.9655 (beat our 0.9645 → confirms undertuning); then re-stack
clears ~0.9670. Run name xgb_strong (does NOT clobber xgb_v1). If flat, ceiling call earned.

Run:  python src/train_xgboost_strong.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgboost import make_sample_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
N_FOLDS = 5
RUN = "xgb_strong"


def inv_balacc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 − balanced_accuracy (lower-is-better, for XGB early stopping = cdeotte's metric)."""
    preds = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
    return 1.0 - balanced_accuracy_score(y_true, preds.astype(int))


XGB_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 7000,
    "learning_rate": 0.012,
    "grow_policy": "lossguide",
    "max_leaves": 72,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": inv_balacc,
    "early_stopping_rounds": 200,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_pd[cat_cols] = oe.fit_transform(train_pd[cat_cols])
        test_pd[cat_cols] = oe.transform(test_pd[cat_cols])
    X, X_test = train_pd[feature_cols], test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    sw = make_sample_weights(y)
    print(f"X {X.shape}  classes {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X.iloc[tri], y[tri], sample_weight=sw[tri],
                  eval_set=[(X.iloc[vai], y[vai])], verbose=False)
        oof[vai] = model.predict_proba(X.iloc[vai])
        test_proba += model.predict_proba(X_test) / N_FOLDS
        best_iters.append(int(getattr(model, "best_iteration", XGB_PARAMS["n_estimators"]) or 0))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}  (best_iter {best_iters[-1]})")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [xgb_v1 0.9645; our LGBM 0.9654; deotte XGB 0.9669]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
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
