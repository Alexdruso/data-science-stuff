"""CatBoost on cdeotte's 240-feature recipe — the 3rd strong base (S6E6).

Reuses deotte_features.build_feature_matrix + the per-fold fold-safe qbin TargetEncoder from
train_xgb_deotte, swapping XGBoost→CatBoost (GPU, symmetric trees → algorithmic diversity from
the XGB base, which the stacker rewards). cdeotte's CatBoost base reports ~0.9697; the FE is the
proven lever (his exact CB hyperparameters not in hand, so a strong-regime config is used).
Same 240 numeric features (model cats ordinal-factorized like the XGB port), inv-freq class
weights, balanced-accuracy early stopping.

GATE: standalone ~0.968+; then add oof_catboost_deotte/test_catboost_deotte to the stack.

Run:  python src/train_catboost_deotte.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES, build_feature_matrix,
)
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import add_fold_safe_te, class_weights, sorted_factorize, te_sources

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED, N_SPLITS = 42, 5
RUN = "catboost_deotte"

CB_PARAMS = {
    "iterations": 5000, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 5.0,
    "loss_function": "MultiClass", "eval_metric": "Accuracy:use_weights=true",
    "task_type": "GPU", "border_count": 254, "random_seed": SEED, "verbose": 0,
}


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
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  model cats {len(MODEL_CAT_COLS)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []
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
        tr_pool = Pool(Xtr, y[tri], weight=sw)
        va_pool = Pool(Xva, y[vai], weight=vw)
        model = CatBoostClassifier(**{**CB_PARAMS, "random_seed": SEED + fold * 100})
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=200)
        oof[vai] = model.predict_proba(Xva)
        test_proba += model.predict_proba(Xte) / N_SPLITS
        best_iters.append(int(model.get_best_iteration() or CB_PARAMS["iterations"]))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  (best_iter {best_iters[-1]})")
        del model, X_tr, X_va, X_te, Xtr, Xva, Xte

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [xgb_deotte 0.96699; realmlp_deotte 0.96888]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
