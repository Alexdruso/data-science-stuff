"""CatBoost on cdeotte's 240-feature recipe WITH NATIVE categorical handling (S6E6).

`train_catboost_deotte.py` factorized the model cats to int codes and then cast the whole
matrix to float32 — feeding CatBoost plain numerics and throwing away its defining strength
(ordered target statistics on categoricals). That version scored 0.9664; cdeotte's CatBoost
(native cats) reports ~0.9697 → a documented +0.003 gap on an existing base.

This variant is a clean A/B: identical FE, identical params, identical fold-safe qbin TE — the
ONLY change is that MODEL_CAT_COLS stay as integer category codes and are passed to the Pool via
`cat_features`, so CatBoost computes ordered TS on them (leak-safe, ordered-boosting internal).
Symmetric oblivious trees + ordered TS = genuine algorithmic diversity from the XGB base.

GATE: standalone OOF argmax must clear catboost_deotte 0.9664 (target ~0.968+); then re-stack and
check whether a STRONG native CatBoost moves the stack (the old 'redundant' verdict was on the weak
0.9664 version sitting inside the envelope). If it beats catboost_deotte, promote → replace it.

Run:  python src/train_catboost_native.py
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
RUN = "catboost_native"

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
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  native cats {len(MODEL_CAT_COLS)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        # Cats stay as INT codes (CatBoost requires int/str for cat_features) — NOT cast to float.
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        cat_idx = [i for i, f in enumerate(feats) if f in MODEL_CAT_COLS]
        if fold == 1:
            print(f"  features used: {len(feats)}/{len(TOP_FEATURES)}  cat_features {len(cat_idx)}")
        # Build dtype-preserving frames: cat cols int32, rest float32.
        num_feats = [f for f in feats if f not in MODEL_CAT_COLS]

        def frame(df: pd.DataFrame) -> pd.DataFrame:
            out = df[feats].copy()
            out[num_feats] = out[num_feats].astype(np.float32)
            for c in MODEL_CAT_COLS:
                out[c] = out[c].astype("int32")
            return out

        Xtr, Xva, Xte = frame(X_tr), frame(X_va), frame(X_te)

        sw, vw = class_weights(y[tri]), class_weights(y[vai])
        tr_pool = Pool(Xtr, y[tri], weight=sw, cat_features=cat_idx)
        va_pool = Pool(Xva, y[vai], weight=vw, cat_features=cat_idx)
        model = CatBoostClassifier(**{**CB_PARAMS, "random_seed": SEED + fold * 100})
        model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=200)
        oof[vai] = model.predict_proba(Xva)
        test_proba += model.predict_proba(Xte) / N_SPLITS
        best_iters.append(int(model.get_best_iteration() or CB_PARAMS["iterations"]))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  (best_iter {best_iters[-1]})")
        del model, X_tr, X_va, X_te, Xtr, Xva, Xte, tr_pool, va_pool

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [catboost_deotte(float-cats) 0.96636; cdeotte 0.9697]")
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
