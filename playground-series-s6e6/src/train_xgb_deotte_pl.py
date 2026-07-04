"""Ensemble pseudo-labeling on xgb_deotte — the highest-ceiling lever (S6E6 campaign).

Our old "PL is dead" verdict used WEAK labelers (single LGBM; RealMLP self-PL). We now have a
~0.970 stack as the TEACHER — a categorically better label source, and the winning
`realmlp-0.96973-ridge-pl` notebook uses PL. Transductive cross-model PL: the strong ensemble
(test_gbdtstack proba) hard-labels high-confidence TEST rows; xgb_deotte is retrained on
[train-fold + high-conf pseudo-test] and predicts the val fold.

⚠️ CV CAVEAT (advisor): the pseudo-test labels come from an ensemble trained on ALL train (incl
each val fold), so this CV is mildly OPTIMISTIC (a diffuse leak through test↔val proximity). PL is
the classic CV-liar → **GATE ON LB, not CV.** Treat any CV gain here as a hypothesis to confirm by
submitting pseudo_xgb_deotte.csv vs xgb_deotte.csv.

Run:  python src/train_xgb_deotte_pl.py
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
from deotte_features import CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES, build_feature_matrix
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import add_fold_safe_te, class_weights, make_xgb_params, sorted_factorize, te_sources

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
PL_CONF, PL_WEIGHT = 0.95, 0.5
TEACHER = "test_gbdtstack.npy"  # strong ensemble test proba (the label source)
RUN = "xgb_deotte_pl"


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    teacher = np.load(RESULTS_DIR / TEACHER)
    pl_conf = teacher.max(axis=1)
    pl_label = teacher.argmax(axis=1)
    pl_mask = pl_conf >= PL_CONF
    print(f"teacher {TEACHER}: {int(pl_mask.sum())}/{len(pl_mask)} test rows ≥{PL_CONF} conf "
          f"({100*pl_mask.mean():.1f}%)  pseudo dist {np.bincount(pl_label[pl_mask])}")

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]

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
        Xtr, Xva, Xte = (X_tr[feats].astype(np.float32).to_numpy(),
                         X_va[feats].astype(np.float32).to_numpy(),
                         X_te[feats].astype(np.float32).to_numpy())

        # append high-conf pseudo-test rows to the training fold
        Xpl, ypl = Xte[pl_mask], pl_label[pl_mask]
        Xtr_full = np.vstack([Xtr, Xpl])
        y_full = np.concatenate([y[tri], ypl])
        w_tr = class_weights(y[tri])
        cw = len(y[tri]) / (len(CLASSES) * np.bincount(y[tri], minlength=len(CLASSES)))
        w_pl = (PL_WEIGHT * cw[ypl]).astype(np.float32)
        w_full = np.concatenate([w_tr, w_pl])

        model = XGBClassifier(**make_xgb_params(SEED + fold * 100))
        model.fit(Xtr_full, y_full, sample_weight=w_full,
                  eval_set=[(Xva, y[vai])], sample_weight_eval_set=[class_weights(y[vai])], verbose=False)
        oof[vai] = model.predict_proba(Xva)
        test_proba += model.predict_proba(Xte) / N_SPLITS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  (+{len(ypl)} pseudo rows)")
        del model, X_tr, X_va, X_te

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [xgb_deotte BASE 0.96699 — CV OPTIMISTIC, gate on LB]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels, id_col=ID_COL)
    print(f"Saved → {RUN}  (submit vs xgb_deotte.csv to LB-gate the PL lever)")


if __name__ == "__main__":
    main()
