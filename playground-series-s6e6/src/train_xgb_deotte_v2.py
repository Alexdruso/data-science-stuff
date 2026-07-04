"""XGB v2: Deotte recipe + 8 top-SHAP experimental features.

Identical to train_xgb_deotte.py except it calls add_experimental_features() after
build_feature_matrix and appends 8 features selected by SHAP rank from train_xgb_extra_fe:

    rs_x_spec_code    rank #4  SHAP 0.187  redshift × spectral_type_ordinal
    u_g_restframe     rank #5  SHAP 0.131  u_g × 1/(1+|z|) — de-redshifted colour
    absmag_g          rank #9  SHAP 0.121  physics: g − dist_mod
    flux_concentration rank #11 SHAP 0.113  flux_std / flux_mean
    inv_1pz           rank #13 SHAP 0.097  cosmological scale factor 1/(1+|z|)
    dist_mod          rank #16 SHAP 0.082  5·log10(|z|) — distance modulus proxy
    absmag_r          rank #20 SHAP 0.058  physics: r − dist_mod
    absmag_u          rank #21 SHAP 0.056  physics: u − dist_mod

Cutoff: SHAP > 0.05 (clear gap to next tier ~0.045).  Everything else kept identical so
OOF comparison to xgb_deotte isolates the feature contribution.

Run:  python src/train_xgb_deotte_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TE_INNER_SPLITS, TE_SMOOTH,
    TOP_FEATURES, add_experimental_features, build_feature_matrix, cat_key,
)
from train_xgb_deotte import add_fold_safe_te, sorted_factorize
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
RUN = "xgb_deotte_v2"

V2_EXTRA_FEATURES = [
    "rs_x_spec_code",     # redshift × spectral_type_ordinal  SHAP 0.187
    "u_g_restframe",      # u_g / (1+|z|)                     SHAP 0.131
    "absmag_g",           # g − dist_mod                       SHAP 0.121
    "flux_concentration", # flux_std / flux_mean               SHAP 0.113
    "inv_1pz",            # 1/(1+|z|)                         SHAP 0.097
    "dist_mod",           # 5·log10(|z|)                      SHAP 0.082
    "absmag_r",           # r − dist_mod                       SHAP 0.058
    "absmag_u",           # u − dist_mod                       SHAP 0.056
]


def balanced_error_metric(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None) -> float:
    yp = y_pred if y_pred.ndim == 2 else y_pred.reshape(-1, len(CLASSES))
    return 1.0 - balanced_accuracy_score(np.asarray(y_true).astype(int), np.argmax(yp, axis=1))


def make_xgb_params(seed: int) -> dict:
    return {
        "objective": "multi:softprob", "num_class": len(CLASSES),
        "eval_metric": balanced_error_metric, "tree_method": "hist", "device": "cuda",
        "learning_rate": 0.012, "n_estimators": 7000, "early_stopping_rounds": 180,
        "max_depth": 0, "max_leaves": 72, "grow_policy": "lossguide", "max_bin": 512,
        "min_child_weight": 10, "gamma": 0.20, "reg_alpha": 0.30, "reg_lambda": 4.0,
        "subsample": 0.82, "colsample_bytree": 0.74, "colsample_bylevel": 0.86,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    wpc = np.float32(len(y)) / (np.float32(len(CLASSES)) * np.maximum(counts, 1.0))
    return wpc[y].astype(np.float32)

def te_sources(top: list[str], cat_cols: list[str]) -> list[str]:
    return [c for c in cat_cols if any(f.startswith(f"TE_{c}_") for f in top)]

def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    X, _ = add_experimental_features(X)
    X_test, _ = add_experimental_features(X_test)

    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    top_set = set(TOP_FEATURES)
    extra_feats = [f for f in V2_EXTRA_FEATURES if f not in top_set]
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  extra feats {len(extra_feats)}: {extra_feats}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []

    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])

        base_feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        feats = base_feats + [f for f in extra_feats if f in X_tr.columns]
        if fold == 1:
            print(f"  features: {len(feats)} ({len(base_feats)} Deotte + {len(feats)-len(base_feats)} extra)")

        Xtr = X_tr[feats].astype(np.float32).to_numpy()
        Xva = X_va[feats].astype(np.float32).to_numpy()
        Xte = X_te[feats].astype(np.float32).to_numpy()

        sw = class_weights(y[tri])
        vw = class_weights(y[vai])
        model = XGBClassifier(**make_xgb_params(SEED + fold * 100))
        model.fit(Xtr, y[tri], sample_weight=sw, eval_set=[(Xva, y[vai])],
                  sample_weight_eval_set=[vw], verbose=False)
        oof[vai] = model.predict_proba(Xva)
        test_proba += model.predict_proba(Xte) / N_SPLITS
        best_iters.append(int(getattr(model, "best_iteration", 0) or 0))
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  (best_iter {best_iters[-1]})")
        del model, X_tr, X_va, X_te, Xtr, Xva, Xte

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [xgb_deotte baseline ~0.96699]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}  mean best_iter {int(np.mean(best_iters))}")
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
