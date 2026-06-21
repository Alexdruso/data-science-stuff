"""All-pairwise binary + Hastie–Tibshirani coupling base — LGBM/CPU (S6E6 diversity lever).

A structural recast in the proven chain_cascade spirit, but DISTINCT in two ways:
  (1) LEARNER: LightGBM-CPU — not currently in the stack in cascade/pairwise form (cascades are
      CatBoost + XGBoost). A new family lever, not a reseed.
  (2) DECOMPOSITION: all THREE one-vs-one binaries (GAL|STAR, GAL|QSO, STAR|QSO), each trained on
      its 2-class subset, then combined SYMMETRICALLY via Hastie–Tibshirani pairwise coupling
      (Classification by Pairwise Coupling, 1998) — vs the cascade's NESTED product factorization
      P(QSO)·P(STAR|STAR∪GAL). The GAL|STAR pair is the documented low-z bottleneck; coupling lets
      the easy QSO pairs sharpen the 3-way posterior differently than the nested cascade does.

Feature pipeline is verbatim xgb_deotte (deotte 240-FE + fold-safe per-class TargetEncoder +
sorted_factorize on model cats), so oof/test arrays are id-sorted and drop into the stack.
Gated on STACK delta via eval_stack_delta.py, NOT standalone.

Run:  python src/train_pairwise_couple.py [--smoke]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES, build_feature_matrix,
)
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import add_fold_safe_te, sorted_factorize, te_sources

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED, N_SPLITS = 42, 5
RUN = "pairwise_couple"
GAL, QSO, STAR = 0, 1, 2
# (pos_class, neg_class): the binary predicts P(pos | pos or neg)
PAIRS = [(STAR, GAL), (QSO, GAL), (STAR, QSO)]


def lgbm_params(seed: int, n_est: int) -> dict:
    dev, gpu_id = get_lgbm_device()
    p = dict(
        objective="binary", n_estimators=n_est, learning_rate=0.03, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=2.0, max_bin=255, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1, device_type=dev,
    )
    if gpu_id >= 0:
        p["gpu_device_id"] = gpu_id
    return p


def couple(r: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """Hastie–Tibshirani pairwise coupling. r[:,i,j] = P(class i | class i or j), r[:,i,i]=0.
    Returns p[:,k] 3-class posteriors (unweighted n_ij=1, the standard form)."""
    n = r.shape[0]
    p = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    eye = np.eye(3, dtype=bool)
    for _ in range(n_iter):
        # mu[:,i,j] = p_i / (p_i + p_j); num_i = Σ_{j≠i} r_ij ; den_i = Σ_{j≠i} mu_ij
        denom = p[:, :, None] + p[:, None, :]
        mu = np.where(eye[None], 0.0, p[:, :, None] / np.clip(denom, 1e-12, None))
        num = r.sum(axis=2)          # Σ_j r_ij  (r_ii=0)
        den = mu.sum(axis=2)         # Σ_j mu_ij
        p = p * num / np.clip(den, 1e-12, None)
        p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p.astype(np.float32)


def build_r(pair_probs: dict[tuple[int, int], np.ndarray], n: int) -> np.ndarray:
    """pair_probs[(pos,neg)] = P(pos | pos or neg) over n rows -> r[:,i,j] matrix."""
    r = np.zeros((n, 3, 3), dtype=np.float64)
    for (pos, neg), pp in pair_probs.items():
        pp = np.clip(pp.astype(np.float64), 1e-6, 1 - 1e-6)
        r[:, pos, neg] = pp
        r[:, neg, pos] = 1.0 - pp
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_est = 400 if args.smoke else 3000

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
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  model cats {len(MODEL_CAT_COLS)}  device {get_lgbm_device()}", flush=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        folds = folds[:1]
    oof = np.zeros((len(X), 3), dtype=np.float32)
    test_proba = np.zeros((len(X_test), 3), dtype=np.float32)
    fold_scores = []

    for fold, (tri, vai) in enumerate(folds, 1):
        print(f"===== fold {fold}/{len(folds)} =====", flush=True)
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in TOP_FEATURES if f in X_tr.columns]
        Xtr = X_tr[feats].astype(np.float32).to_numpy()
        Xva = X_va[feats].astype(np.float32).to_numpy()
        Xte = X_te[feats].astype(np.float32).to_numpy()
        ytr = y[tri]

        val_probs, test_probs = {}, {}
        for pos, neg in PAIRS:
            m = (ytr == pos) | (ytr == neg)
            yb = (ytr[m] == pos).astype(np.int32)
            # internal val subset for early stopping = val rows of these two classes
            mv = (y[vai] == pos) | (y[vai] == neg)
            model = LGBMClassifier(**lgbm_params(SEED + fold * 10 + pos, n_est))
            model.fit(Xtr[m], yb, eval_set=[(Xva[mv], (y[vai][mv] == pos).astype(np.int32))],
                      eval_metric="auc",
                      callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
            val_probs[(pos, neg)] = model.predict_proba(Xva)[:, 1]   # ALL val rows
            test_probs[(pos, neg)] = model.predict_proba(Xte)[:, 1]
            print(f"  pair P({CLASSES[pos]}|{CLASSES[pos]},{CLASSES[neg]}) best_iter {model.best_iteration_}", flush=True)
            del model

        oof[vai] = couple(build_r(val_probs, len(vai)))
        test_proba += couple(build_r(test_probs, len(X_test))) / len(folds)
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        rec_f = recall_score(y[vai], np.argmax(oof[vai], axis=1), average=None, labels=[0, 1, 2])
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  recall(G,Q,S) {rec_f.round(4)}", flush=True)
        del X_tr, X_va, X_te, Xtr, Xva, Xte

    if args.smoke:
        print(f"\n[SMOKE] fold-1 balanced_acc {fold_scores[0]:.5f} — no save.")
        return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
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
