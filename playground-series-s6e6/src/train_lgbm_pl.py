"""LGBM + transductive PSEUDO-LABELING — the discriminating test (PS S6E6).

cdeotte's 0.9702-CV stack is built from base models individually at LB 0.968-0.9698
— including plain GBDTs (his CatBoost 0.9697, XGB 0.968) that beat ours (0.9627,
0.9645) by +0.003-0.007. No feature engineering, no special stacker magic (the LR
meta adds only ~0.001). The shared ingredient that lifts even vanilla boosters is
most likely pseudo-labeling: the 247k test rows are in-distribution (adversarial
AUC=0.5), so adding them with predicted labels ~doubles training data with the
exact vectors we're scored on (transductive).

This is a DISCRIMINATING experiment on our 0.9654 LGBM:
  +0.004-0.007 → PL is the lever → roll out across models + stack.
  only ~+0.001 → PL is NOT it → the recipe lives in a base notebook we haven't seen.

CV-honesty guardrail (advisor): pseudo-labels are generated PER FOLD, inside the
loop — base model fit on the train-fold pseudo-labels TEST only; the retrain adds
those pseudo-test rows; the pure val-fold (real labels) is never pseudo-labeled and
is what OOF scores. No full-data PL reused across folds (that would leak).

Run:  python src/train_lgbm_pl.py
"""

import json
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
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 5
PL_CONF = 0.90      # add test rows whose base max-proba ≥ this as hard pseudo-labels
PL_WEIGHT = 0.5     # downweight pseudo rows vs real train rows
RUN = "lgbm_pl"


def base_params() -> dict[str, object]:
    dev_type, n_jobs = get_lgbm_device()
    p: dict[str, object] = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "n_estimators": 1000, "class_weight": "balanced", "random_state": 42,
        "verbose": -1, "n_jobs": n_jobs, "device_type": dev_type,
    }
    bp = json.loads((RESULTS_DIR / "best_params.json").read_text())
    p.update(bp)  # tuned shape (lr=0.046, num_leaves=173, ...) — the 0.9654 config
    return p


def main() -> None:
    params = base_params()
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")
        test_pd[col] = test_pd[col].astype("category")
    X, X_test = train_pd[feature_cols], test_pd[feature_cols]
    test_ids = test_pd["id"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    print(f"X {X.shape}  PL_CONF={PL_CONF} PL_WEIGHT={PL_WEIGHT}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))
    fold_scores, n_pl = [], []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        # round 0: base model on train-fold → pseudo-label TEST only
        m0 = LGBMClassifier(**params)
        m0.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
               callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        tp0 = m0.predict_proba(X_test)
        conf = tp0.max(axis=1) >= PL_CONF
        pl_y = tp0[conf].argmax(axis=1)
        n_pl.append(int(conf.sum()))

        # round 1: retrain on train-fold ⋃ confident pseudo-test; predict PURE val
        X_aug = pd.concat([X.iloc[tri], X_test.iloc[conf.nonzero()[0]]], ignore_index=True)
        y_aug = np.concatenate([y[tri], pl_y])
        w = np.concatenate([np.ones(len(tri)), np.full(int(conf.sum()), PL_WEIGHT)])
        m1 = LGBMClassifier(**params)
        m1.fit(X_aug, y_aug, sample_weight=w)
        oof[vai] = m1.predict_proba(X.iloc[vai])
        test_proba += m1.predict_proba(X_test) / N_FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}  (pseudo rows {n_pl[-1]})")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [incumbent lgbm 0.9654]  "
          f"avg pseudo rows {int(np.mean(n_pl))}/{len(X_test)}")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  "
          f"[incumbent STAR 0.963]")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}  [incumbent 0.9657]")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
