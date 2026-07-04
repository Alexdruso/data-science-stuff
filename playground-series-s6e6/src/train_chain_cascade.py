"""Classifier-chain CASCADE base — QSO -> STAR/GAL (S6E6).

A new diverse base that factors P(class|x) as a product of conditionals instead of one direct
3-way softmax (the decorrelation source the re-FE experiments lacked):

    Stage 1  p_qso       = P(QSO  | x)                       binary, ALL train rows
    Stage 2  p_star_cond = P(STAR | x, star-or-galaxy)       binary, STAR+GAL train rows only

    P(QSO)    = p_qso
    P(STAR)   = (1 - p_qso) * p_star_cond
    P(GALAXY) = (1 - p_qso) * (1 - p_star_cond)              (sums to 1 by construction)

Stage 2 is a dedicated full-capacity, full-feature binary on the documented low-z STAR/GALAXY
bottleneck — NOT the closed low-z specialist (that was crippled to z<0.25 + 14 features and was
strictly out-ranked by the stack). Class order matches LabelEncoder GALAXY0/QSO1/STAR2.

Gated on STACK delta, not standalone. A built-in operating-point-free ranking pre-check (the exact
test that closed the specialist, lrstack low-z AUC 0.98824) guards against the standalone mirage.

Run:  python src/train_chain_cascade.py [--learner catboost|xgboost] [--smoke]
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED, N_SPLITS = 42, 5
# RUN is derived from the learner in main(): catboost -> "chain_cascade" (preserves the live
# 10-base stack files), xgboost -> "chain_cascade_xgb" (a distinct, decorrelated second cascade).
RUN_BY_LEARNER = {"catboost": "chain_cascade", "xgboost": "chain_cascade_xgb"}
CLASSES = ["GALAXY", "QSO", "STAR"]  # LabelEncoder order
GAL, QSO, STAR = 0, 1, 2
ORIGINAL_WEIGHT = 0.06
PREDICT_BATCH = 80_000
STACK_LOWZ_AUC = 0.98824  # lrstack low-z STAR-vs-GAL ranking AUC (the bar that closed the specialist)
LOWZ_THRESH = 0.25


def combine(p_qso: np.ndarray, p_star_cond: np.ndarray) -> np.ndarray:
    """(n,) Stage-1 P(QSO) and (n,) Stage-2 P(STAR|SG) -> (n,3) GALAXY/QSO/STAR probs."""
    rest = 1.0 - p_qso
    out = np.empty((len(p_qso), 3), dtype=np.float32)
    out[:, GAL] = rest * (1.0 - p_star_cond)
    out[:, QSO] = p_qso
    out[:, STAR] = rest * p_star_cond
    return out


# --------------------------------------------------------------------------------------------
# CatBoost cascade — native cat_features on the v3 cat-heavy FE (the ordered-TS lever)
# --------------------------------------------------------------------------------------------
def run_catboost(smoke: bool):
    from catboost import CatBoostClassifier, Pool

    from catboost_v3_features import build_features_cat_v3

    iters = 300 if smoke else 5000
    es = 60 if smoke else 260

    tr = pl.read_csv(DATA_DIR / "train.csv").sort("id").to_pandas()
    te = pl.read_csv(DATA_DIR / "test.csv").sort("id").to_pandas()
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.with_columns(pl.col("class").str.to_uppercase()).filter(
        pl.col("class").is_in(CLASSES)).to_pandas()

    cls_to_int = {c: i for i, c in enumerate(CLASSES)}
    y = tr["class"].map(cls_to_int).astype("int64").to_numpy()
    y_orig = orig["class"].map(cls_to_int).astype("int64").to_numpy()
    z = tr["redshift"].to_numpy(np.float32)
    test_ids = te["id"].to_numpy()

    print(f"building v3 cat features (train {len(tr)} / test {len(te)} / orig {len(orig)})...", flush=True)
    X, X_test, X_orig, cat_cols = build_features_cat_v3(tr, te, orig)
    print(f"X={X.shape}  cats={len(cat_cols)}", flush=True)

    def params(seed):
        return dict(
            loss_function="Logloss", eval_metric="AUC", iterations=iters, depth=8,
            learning_rate=0.042, l2_leaf_reg=8.0, random_strength=1.2,
            bootstrap_type="Bayesian", bagging_temperature=0.2, one_hot_max_size=16,
            max_ctr_complexity=3, auto_class_weights="Balanced", border_count=254,
            random_seed=seed, early_stopping_rounds=es, task_type="GPU", devices="0",
            gpu_ram_part=0.85, gpu_cat_features_storage="CpuPinnedMemory", thread_count=4,
            allow_writing_files=False, verbose=250,
        )

    def predict_pstar(model, Xd):  # batched P(positive)
        parts = []
        for s in range(0, len(Xd), PREDICT_BATCH):
            pool = Pool(Xd.iloc[s:s + PREDICT_BATCH], cat_features=cat_cols)
            parts.append(model.predict_proba(pool)[:, 1].astype("float32"))
            del pool
            gc.collect()
        return np.concatenate(parts)

    def fit_binary(X_fit, yb_fit, w_fit, X_val, yb_val, seed):
        train_pool = Pool(X_fit, yb_fit, cat_features=cat_cols, weight=w_fit)
        valid_pool = Pool(X_val, yb_val, cat_features=cat_cols)
        model = CatBoostClassifier(**params(seed))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        del train_pool, valid_pool
        return model

    oof = np.zeros((len(X), 3), dtype="float32")
    p_star_oof = np.zeros(len(X), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")
    sg_orig = (y_orig == GAL) | (y_orig == STAR)

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        print(f"===== fold {fold}/{len(folds)} =====", flush=True)
        # Stage 1: QSO-vs-rest on ALL train rows (+ original rows @0.06)
        X1 = pd.concat([X.iloc[tri], X_orig], axis=0, ignore_index=True)
        yb1 = np.concatenate([(y[tri] == QSO).astype(int), (y_orig == QSO).astype(int)])
        w1 = np.ones(len(yb1), dtype="float32"); w1[len(tri):] = ORIGINAL_WEIGHT
        m1 = fit_binary(X1, yb1, w1, X.iloc[vai], (y[vai] == QSO).astype(int), SEED + fold)
        p_qso_val = predict_pstar(m1, X.iloc[vai])
        if not smoke:
            p_qso_test = predict_pstar(m1, X_test)
        del m1, X1; gc.collect()

        # Stage 2: STAR-vs-GAL on STAR+GAL train rows only (+ original STAR/GAL rows @0.06)
        tri_sg = tri[(y[tri] == GAL) | (y[tri] == STAR)]
        X2 = pd.concat([X.iloc[tri_sg], X_orig[sg_orig]], axis=0, ignore_index=True)
        yb2 = np.concatenate([(y[tri_sg] == STAR).astype(int), (y_orig[sg_orig] == STAR).astype(int)])
        w2 = np.ones(len(yb2), dtype="float32"); w2[len(tri_sg):] = ORIGINAL_WEIGHT
        vai_sg = vai[(y[vai] == GAL) | (y[vai] == STAR)]
        m2 = fit_binary(X2, yb2, w2, X.iloc[vai_sg], (y[vai_sg] == STAR).astype(int), SEED + fold)
        p_star_val = predict_pstar(m2, X.iloc[vai])  # all val rows (QSO down-weighted by 1-p_qso)
        if not smoke:
            p_star_test = predict_pstar(m2, X_test)
        del m2, X2; gc.collect()

        oof[vai] = combine(p_qso_val, p_star_val)
        p_star_oof[vai] = p_star_val
        if not smoke:
            test_pred += combine(p_qso_test, p_star_test) / N_SPLITS
        sc = balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))
        print(f"fold {fold} bal-acc {sc:.6f}", flush=True)
        gc.collect()

    return oof, test_pred, p_star_oof, y, z, test_ids


# --------------------------------------------------------------------------------------------
# XGBoost cascade — deotte 240-FE + fold-safe per-class TargetEncoder (verbatim xgb_deotte pattern)
# --------------------------------------------------------------------------------------------
def run_xgboost(smoke: bool):
    from xgboost import XGBClassifier

    from deotte_features import (
        CLASS_TO_INT, ID_COL, TARGET, TOP_FEATURES, build_feature_matrix,
    )
    from train_xgb_deotte import add_fold_safe_te, sorted_factorize, te_sources

    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID_COL).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID_COL).reset_index(drop=True)
    orig = pd.read_csv(DATA_DIR / "star_classification.csv")
    orig = orig[orig["u"] > -1000.0].reset_index(drop=True)
    y = train[TARGET].map(CLASS_TO_INT).astype("int64").to_numpy()
    y_orig = orig[TARGET].map(CLASS_TO_INT).astype("int64").to_numpy()
    z = train["redshift"].to_numpy(np.float32)
    test_ids = test[ID_COL].to_numpy()

    X, X_test, cat_cols = build_feature_matrix(train, test, orig, y_orig)
    TE_COLS = te_sources(TOP_FEATURES, cat_cols)
    MODEL_CAT_COLS = [c for c in cat_cols if c in TOP_FEATURES]
    print(f"X {X.shape}  TE sources {len(TE_COLS)}  model cats {len(MODEL_CAT_COLS)}", flush=True)

    n_est = 400 if smoke else 7000
    es = 60 if smoke else 180

    def params(seed):
        return dict(
            objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
            learning_rate=0.012, n_estimators=n_est, early_stopping_rounds=es,
            max_depth=0, max_leaves=72, grow_policy="lossguide", max_bin=512,
            min_child_weight=10, gamma=0.20, reg_alpha=0.30, reg_lambda=4.0,
            subsample=0.82, colsample_bytree=0.74, colsample_bylevel=0.86,
            random_state=seed, n_jobs=4, verbosity=0,
        )

    def invfreq_w(yb):
        c = np.bincount(yb, minlength=2).astype(np.float32)
        w = np.float32(len(yb)) / (2.0 * np.maximum(c, 1.0))
        return w[yb].astype(np.float32)

    oof = np.zeros((len(X), 3), dtype="float32")
    p_star_oof = np.zeros(len(X), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if smoke:
        folds = folds[:1]

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

        # Stage 1: QSO-vs-rest, all rows
        yb1 = (y[tri] == QSO).astype(int)
        m1 = XGBClassifier(**params(SEED + fold * 100))
        m1.fit(Xtr, yb1, sample_weight=invfreq_w(yb1),
               eval_set=[(Xva, (y[vai] == QSO).astype(int))], verbose=False)
        p_qso_val = m1.predict_proba(Xva)[:, 1].astype("float32")
        p_qso_test = m1.predict_proba(Xte)[:, 1].astype("float32")

        # Stage 2: STAR-vs-GAL, STAR+GAL rows only
        sg = (y[tri] == GAL) | (y[tri] == STAR)
        va_sg = (y[vai] == GAL) | (y[vai] == STAR)
        yb2 = (y[tri][sg] == STAR).astype(int)
        m2 = XGBClassifier(**params(SEED + fold * 100))
        m2.fit(Xtr[sg], yb2, sample_weight=invfreq_w(yb2),
               eval_set=[(Xva[va_sg], (y[vai][va_sg] == STAR).astype(int))], verbose=False)
        p_star_val = m2.predict_proba(Xva)[:, 1].astype("float32")
        p_star_test = m2.predict_proba(Xte)[:, 1].astype("float32")

        oof[vai] = combine(p_qso_val, p_star_val)
        p_star_oof[vai] = p_star_val
        test_pred += combine(p_qso_test, p_star_test) / N_SPLITS
        sc = balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))
        print(f"fold {fold} bal-acc {sc:.6f}", flush=True)
        del m1, m2, X_tr, X_va, X_te, Xtr, Xva, Xte
        gc.collect()

    return oof, test_pred, p_star_oof, y, z, test_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", choices=["catboost", "xgboost"], default="catboost")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run = RUN_BY_LEARNER[args.learner]
    runner = run_catboost if args.learner == "catboost" else run_xgboost
    oof, test_pred, p_star_oof, y, z, test_ids = runner(args.smoke)

    if args.smoke:
        nz = np.where(oof.sum(axis=1) > 0)[0]
        print(f"smoke OK ({len(nz)} val rows; sum-to-1 check: {float(oof[nz[0]].sum()):.4f})")
        return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"\n{run} ({args.learner}) OOF argmax {argmax:.5f}  thresh {best:.5f}  "
          f"[catboost_v3 0.96891 / realmlp_deotte 0.96888]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")

    # operating-point-free ranking pre-check on low-z true-{STAR,GAL} OOF rows (the specialist-killer)
    mask = (z < LOWZ_THRESH) & ((y == GAL) | (y == STAR))
    lowz_auc = roc_auc_score((y[mask] == STAR).astype(int), p_star_oof[mask])
    verdict = "GREEN — out-ranks stack" if lowz_auc > STACK_LOWZ_AUC else "below stack (not a kill; gate on stack delta)"
    print(f"low-z STAR-vs-GAL ranking AUC {lowz_auc:.5f}  vs lrstack {STACK_LOWZ_AUC}  -> {verdict}")

    np.save(RESULTS_DIR / f"oof_{run}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run}.npy", test_pred)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{run}.json")
    save_cv_result(RESULTS_DIR, run, [], best, metric_name="balanced_acc")
    labels = [CLASSES[i] for i in np.argmax(test_pred * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{run}.csv", test_ids, "class", labels)
    print(f"Saved -> {run} (oof/test/threshold/submission). Next: add '{run}' to "
          f"build_lr_stack + build_gbdt_stack MODELS and re-run; gate on stack argmax vs "
          f"lrstack 0.97055 / metablend 0.97061. Also check the stacker assigns it nonzero weight.")


if __name__ == "__main__":
    main()
