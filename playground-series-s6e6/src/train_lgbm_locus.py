"""LGBM + stellar-locus features (class-conditional color-manifold likelihood).

Web-scouted lever: top S6E6 solutions use astrophysical "stellar locus" feature
engineering. Stars trace a tight CURVED 1-D locus in color-color space; galaxies/
QSOs scatter off it. Axis-aligned GBDT splits approximate that curve poorly (why
raw bands plateau at ~0.965). We hand the model the manifold structure directly:
per CV fold, fit a Gaussian-mixture density per class on the colors+redshift of the
fold's TRAIN rows, then score every row's log-likelihood under each class density →
3 features ("how GALAXY/QSO/STAR-like is this object's color manifold position").

Leakage-safe like target encoding (CLAUDE.md Rule 2): the GMMs see only the fold's
train split; val/test are scored, never fitted. Test features are averaged across
the 5 folds. Strong-v2 LGBM config; gate argmax vs 0.9654.

Run:  python src/train_lgbm_locus.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights
from train_lgbm_strong import balanced_acc_eval

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 5
LR, NUM_LEAVES, N_ESTIMATORS, PATIENCE = 0.02, 63, 20000, 200
RUN = "lgbm_locus_color"
# Canonical stellar locus is COLOR-ONLY (no redshift) — its whole point is
# separating stars from extragalactic objects orthogonally to redshift, which the
# GBDT already splits on. v1 wrongly included redshift → logliks collapsed onto a
# smooth restatement of redshift and added nothing. These 4 adjacent SDSS colors
# are the independent locus coords.
LOCUS_COLS = ["u_g", "g_r", "r_i", "i_z"]
N_COMPONENTS = 12  # mixture components — enough to trace the curved color locus


def locus_loglik(
    M_tr: np.ndarray, y_tr: np.ndarray, *Ms: np.ndarray
) -> list[np.ndarray]:
    """Fit a GMM per class on (M_tr, y_tr); return per-class loglik columns for
    each matrix in Ms (each → array of shape n×3)."""
    sc = StandardScaler().fit(M_tr)
    gmms = {
        c: GaussianMixture(N_COMPONENTS, covariance_type="full", random_state=42,
                           reg_covar=1e-4).fit(sc.transform(M_tr[y_tr == c]))
        for c in (0, 1, 2)
    }
    out = []
    for M in Ms:
        Ms_ = sc.transform(M)
        out.append(np.column_stack([gmms[c].score_samples(Ms_) for c in (0, 1, 2)]))
    return out


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    params: dict[str, object] = {
        "objective": "multiclass", "num_class": 3, "metric": "None",
        "class_weight": "balanced", "random_state": 42, "verbose": -1,
        "n_jobs": n_jobs, "device_type": dev_type,
    }
    bp = json.loads((RESULTS_DIR / "best_params.json").read_text())
    for k in ("learning_rate", "num_leaves"):
        bp.pop(k, None)
    params.update(bp)
    params.update(learning_rate=LR, num_leaves=NUM_LEAVES, n_estimators=N_ESTIMATORS)

    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    base_feats = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    locus_names = ["loglik_GAL", "loglik_QSO", "loglik_STAR"]
    feature_cols = base_feats + locus_names

    train_pd, test_pd = train_pl.to_pandas(), test_pl.to_pandas()
    for col in cat_cols:
        train_pd[col] = train_pd[col].astype("category")
        test_pd[col] = test_pd[col].astype("category")
    le = LabelEncoder()
    y = le.fit_transform(train_pd[TARGET].to_numpy())
    test_ids = test_pd["id"].to_numpy()
    M_all = train_pd[LOCUS_COLS].to_numpy()
    M_test = test_pd[LOCUS_COLS].to_numpy()
    print(f"base {len(base_feats)} + locus 3 | GMM {N_COMPONENTS}c on {LOCUS_COLS} "
          f"device={dev_type}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(train_pd), 3))
    test_proba = np.zeros((len(test_pd), 3))
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(train_pd, y), 1):
        # fold-aware locus features: GMMs fit on TRAIN split only
        F_tr, F_val, F_test = locus_loglik(M_all[tri], y[tri], M_all[tri], M_all[vai], M_test)
        X_tr = train_pd.iloc[tri][base_feats].copy()
        X_val = train_pd.iloc[vai][base_feats].copy()
        X_te = test_pd[base_feats].copy()
        for j, nm in enumerate(locus_names):
            X_tr[nm], X_val[nm], X_te[nm] = F_tr[:, j], F_val[:, j], F_test[:, j]

        m = LGBMClassifier(**params)
        m.fit(X_tr, y[tri], eval_set=[(X_val, y[vai])], eval_metric=balanced_acc_eval,
              callbacks=[early_stopping(PATIENCE, first_metric_only=True, verbose=False),
                         log_evaluation(0)])
        oof[vai] = m.predict_proba(X_val)
        test_proba += m.predict_proba(X_te) / N_FOLDS
        best_iters.append(m.best_iteration_ or N_ESTIMATORS)
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}  (best_iter {best_iters[-1]})")

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [incumbent 0.9654]  "
          f"mean best_iter {int(np.mean(best_iters))}")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}")
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
