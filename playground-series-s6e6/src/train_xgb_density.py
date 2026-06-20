"""Density-augmented XGB: cdeotte-240-FE + 3 flow per-class log-densities (S6E6 idea #2).

The class-conditional normalizing flow (train_flow.py) is genuinely orthogonal (errs where class
DENSITIES overlap, not where a margin is thin) but too WEAK standalone (0.93) → the LR meta
down-weights it to ~0 and +flow is flat. This feeds the flow's 3 raw log-densities
[log p(x|GAL/QSO/STAR)] as FEATURES into a strong discriminative XGB instead. Rationale: a learned
bijection's densities are NOT a deterministic function of the existing columns (unlike physics FE),
so a GBDT *might* extract orthogonal density×feature interactions the stacker never saw.

⚠️ This collides with two of our own findings: (1) "deterministic fn of existing cols → GBDT
already extracts → flat", (2) "flow logits ≈ density-ratios the stacker already saw". The ONLY
escape is a genuine density×deotte-feature interaction. So GATE CHEAP: fold-1 prints the 3 density
features' XGB importance. If ~0 (like the lineage features were), this is dead — STOP, don't run
5 folds + stack. Only continue if they carry real importance.

PREREQ: run `python -u src/train_flow.py --save-logdens` first to create
results/oof_flow_logdens.npy + results/test_flow_logdens.npy (raw log p(x|class), OOF/leak-safe,
id-sorted = aligned with the deotte matrix; SORT_KEY==["id"]).

Run:  python -u src/train_xgb_density.py
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
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES,
    build_feature_matrix,
)
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import (
    add_fold_safe_te, balanced_error_metric, class_weights, make_xgb_params,
    sorted_factorize, te_sources,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED, N_SPLITS = 42, 5
RUN = "xgb_density"
DENS_COLS = [f"flow_logdens_{c}" for c in CLASSES]


def main() -> None:
    logdens_tr_path = RESULTS_DIR / "oof_flow_logdens.npy"
    logdens_te_path = RESULTS_DIR / "test_flow_logdens.npy"
    if not logdens_tr_path.exists() or not logdens_te_path.exists():
        raise FileNotFoundError(
            "Missing flow log-densities. Run: python -u src/train_flow.py --save-logdens"
        )

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype(np.int64).to_numpy()
    test_ids = test[ID_COL].to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)

    # append the 3 flow log-density columns (OOF train / fold-averaged test; id-sorted = aligned)
    ld_tr = np.load(logdens_tr_path).astype(np.float32)
    ld_te = np.load(logdens_te_path).astype(np.float32)
    assert ld_tr.shape == (len(X), 3) and ld_te.shape == (len(X_test), 3), \
        f"logdens shape mismatch: {ld_tr.shape} vs X {X.shape}"
    for j, col in enumerate(DENS_COLS):
        X[col] = ld_tr[:, j]
        X_test[col] = ld_te[:, j]
    feature_list = list(TOP_FEATURES) + DENS_COLS

    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    print(f"X {X.shape}  +{len(DENS_COLS)} density cols  TE sources {len(TE_COLS)}  cats {len(MODEL_CAT_COLS)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    fold_scores, best_iters = [], []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
        for c in MODEL_CAT_COLS:
            X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
        feats = [f for f in feature_list if f in X_tr.columns]
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

        if fold == 1:
            # ★ CHEAP GATE: are the 3 density features actually used by XGB?
            imp = model.feature_importances_
            name_to_imp = dict(zip(feats, imp))
            dens_imp = {c: float(name_to_imp.get(c, 0.0)) for c in DENS_COLS}
            ranked = sorted(name_to_imp.items(), key=lambda x: -x[1])
            dens_ranks = {c: [r for r, (n, _) in enumerate(ranked) if n == c][0]
                          for c in DENS_COLS if c in name_to_imp}
            print(f"  [GATE fold-1] density importances {dens_imp}")
            print(f"  [GATE fold-1] density ranks (of {len(feats)}): {dens_ranks}")
            total_dens = sum(dens_imp.values())
            print(f"  [GATE fold-1] density total importance {total_dens:.5f}  "
                  f"(if ~0 like lineage → DEAD, stop after this run)")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [xgb_deotte 0.96699; gate = eval_stack_delta, NOT this]")
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
    print(f"Saved → {RUN}  (next: eval_stack_delta with xgb_density added)")


if __name__ == "__main__":
    main()
