"""XGB on Deotte features + 40 experimental transforms, with SHAP importance after fold 1.

Goal: discover which of our 40 hand-crafted nonlinear features carry real signal on top of
the ~240-feature Deotte matrix when an ACTUAL GBDT (not LR) sees them.  SHAP on the fold-1
validation set gives true marginal importance, not a proxy.

Differences from train_xgb_deotte.py:
  - Calls add_experimental_features() after build_feature_matrix; appends ~40 new columns.
  - Feature set = TOP_FEATURES + experimental features (deduped).
  - After fold 1 finishes, runs shap.TreeExplainer on a 5 000-row val sample and prints
    the top-40 features ranked by mean |SHAP| across classes.  Saves the bar chart to
    results/shap_xgb_extra_fe.png.
  - oof/test/submission saved as xgb_extra_fe (separate from xgb_deotte so stacker can compare).

Run:  python src/train_xgb_extra_fe.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, EXPERIMENTAL_FEATURES, ID_COL, INT_TO_CLASS,
    TARGET, TE_INNER_SPLITS, TE_SMOOTH, TOP_FEATURES,
    add_experimental_features, build_feature_matrix, cat_key,
)
from train_xgb_deotte import add_fold_safe_te, sorted_factorize
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
RUN = "xgb_extra_fe"
SHAP_SAMPLE = 5_000


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

def plot_shap_importance(
    shap_vals: np.ndarray, feature_names: list[str], out_path: Path, top_n: int = 40
) -> None:
    # shap_vals: (n_samples, n_features, n_classes) or list-of-arrays
    if isinstance(shap_vals, list):
        arr = np.stack(shap_vals, axis=2)
    else:
        arr = shap_vals
    mean_abs = np.abs(arr).mean(axis=(0, 2))     # (n_features,)
    idx = np.argsort(mean_abs)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(range(len(names))[::-1], vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP| (all classes)")
    ax.set_title(f"XGB extra-FE — top {top_n} features by SHAP (fold 1 val, {SHAP_SAMPLE} rows)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  SHAP chart saved → {out_path}")


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    # Build Deotte matrix then augment with experimental features
    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    X, exp_train = add_experimental_features(X)
    X_test, _ = add_experimental_features(X_test)
    print(f"Deotte matrix: {X.shape}  experimental features added: {len(exp_train)}")

    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    top_set = set(TOP_FEATURES)
    EXTRA_FEATS = [f for f in EXPERIMENTAL_FEATURES if f not in top_set]
    print(f"TE sources: {len(TE_COLS)}  model cats: {len(MODEL_CAT_COLS)}  "
          f"extra numeric feats: {len(EXTRA_FEATS)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []
    shap_done = False

    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])

        base_feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        extra_feats = [f for f in EXTRA_FEATS if f in X_tr.columns]
        feats = base_feats + extra_feats
        if fold == 1:
            print(f"  features: {len(feats)} total "
                  f"({len(base_feats)} Deotte + {len(extra_feats)} experimental)")

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

        if fold == 1 and not shap_done:
            print(f"  Computing SHAP on {SHAP_SAMPLE}-row val sample …")
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(vai), min(SHAP_SAMPLE, len(vai)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xva[idx])
            # shap_vals: (n_samples, n_feats, n_classes) for XGB multiclass
            if isinstance(shap_vals, list):
                sv_arr = np.stack(shap_vals, axis=2)
            else:
                sv_arr = np.asarray(shap_vals)
            mean_abs = np.abs(sv_arr).mean(axis=(0, 2))
            ranked = np.argsort(mean_abs)[::-1]
            print("\n  Top-50 features by mean |SHAP| (all classes):")
            print(f"  {'rank':>4}  {'feature':<45}  {'mean|SHAP|':>10}  {'exp?':>5}")
            print(f"  {'-'*4}  {'-'*45}  {'-'*10}  {'-'*5}")
            extra_set = set(EXTRA_FEATS)
            for rank, fi in enumerate(ranked[:50], 1):
                tag = "✓" if feats[fi] in extra_set else ""
                print(f"  {rank:>4}  {feats[fi]:<45}  {mean_abs[fi]:>10.5f}  {tag:>5}")
            plot_shap_importance(sv_arr, feats,
                                 RESULTS_DIR / "shap_xgb_extra_fe.png", top_n=50)
            del explainer, shap_vals, sv_arr
            shap_done = True

        del model, X_tr, X_va, X_te, Xtr, Xva, Xte

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [baseline xgb_deotte ~0.96699]")
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
