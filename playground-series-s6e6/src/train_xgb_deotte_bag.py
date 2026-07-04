"""Seed-bagged xgb_deotte — variance reduction on our highest-variance strong base (S6E6).

train_xgb_deotte.py fits ONE seed per fold. With 72-leaf lossguide + subsample 0.82 /
colsample 0.74, that base carries real per-seed boundary noise (fold scores swing 0.966–0.968).
LGBM seed-bagging was flat here because our LGBM is low-variance/tuned-stable — but this XGB is a
higher-capacity, heavily-subsampled model, so averaging seeds should both lift standalone a hair
and hand the stacker a SMOOTHER OOF (its actual consumable). Everything else is identical to
train_xgb_deotte (same FE, fold-safe qbin TE, params, early-stop) — the only change is averaging
N_SEEDS XGB fits per fold.

GATE: standalone vs xgb_deotte 0.96699; then SWAP xgb_deotte→xgb_deotte_bag in the stack MODELS
(collinear — don't keep both) and re-run build_lr_stack / build_gbdt_stack.

Run:  python src/train_xgb_deotte_bag.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES, build_feature_matrix,
)
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import (
    add_fold_safe_te, class_weights, make_xgb_params, sorted_factorize, te_sources,
)

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
N_SEEDS = 3
SEEDS = [42, 2024, 7]
RUN = "xgb_deotte_bag"


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  model cats {len(MODEL_CAT_COLS)}  seeds {SEEDS}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        if fold == 1:
            print(f"  features used: {len(feats)}/{len(TOP_FEATURES)}")
        Xtr = X_tr[feats].astype(np.float32).to_numpy()
        Xva = X_va[feats].astype(np.float32).to_numpy()
        Xte = X_te[feats].astype(np.float32).to_numpy()
        sw, vw = class_weights(y[tri]), class_weights(y[vai])

        va_acc = np.zeros((len(vai), len(CLASSES)), dtype=np.float64)
        te_acc = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
        for s in SEEDS:
            params = make_xgb_params(s + fold * 100)
            model = XGBClassifier(**params)
            model.fit(Xtr, y[tri], sample_weight=sw, eval_set=[(Xva, y[vai])],
                      sample_weight_eval_set=[vw], verbose=False)
            va_acc += model.predict_proba(Xva)
            te_acc += model.predict_proba(Xte)
            del model
        oof[vai] = (va_acc / N_SEEDS).astype(np.float32)
        test_proba += (te_acc / N_SEEDS).astype(np.float32) / N_SPLITS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}")
        del X_tr, X_va, X_te, Xtr, Xva, Xte

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [xgb_deotte(single seed) 0.96699]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels, id_col=ID_COL)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
