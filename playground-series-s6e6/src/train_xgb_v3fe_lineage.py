"""xgb_v3fe + ALPHA-LINEAGE features (PS S6E6).

EDA (notebooks/eda_error_analysis.py + this session's transfer test) found a synthetic-
generator lineage artifact: 31% of train+test rows share an EXACT `alpha` float value, and
29.5% of TEST rows fall in an alpha-group that also appears in train. A held-out A→B transfer
test confirms the group→class signal is REAL but weak and noisy: a STAR-leaning alpha group
predicts STAR ~2.8x the base rate, a GALAXY-leaning group is 71% GALAXY (vs 65% prior) — the
latter is the interesting one, it can rescue the GAL→STAR low-z errors that are the metric
bottleneck (GALAXY recall 0.9606). The signal is INVISIBLE to the existing models (exact alpha
floats → trees split ranges, never the per-value class distribution).

Lever: fold-safe smoothed per-class target-encoding of the exact-alpha value, RESTRICTED to
test-present alphas (the rest sentinel-bucketed to the global prior). The restriction is what
makes OOF a faithful LB predictor — it kills the train-only-group inflation (a group with no
test rows can only manufacture OOF gain that never transfers). 3 continuous rate cols + log
group-size (train+test) bolted onto the strong xgb_v3fe matrix. NOT the orig-match one-hot
(agreement 0.506 < always-GALAXY 0.654 → below baseline, dropped).

GATE: standalone OOF argmax (vs xgb_v3fe 0.96711) AND — the binding test — a positive STACK
delta over the current 11-base lrstack 0.970481. <0.0014 = noise.

Run:  python src/train_xgb_v3fe_lineage.py [--smoke]
"""

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from catboost_v3_features import CLASSES, build_features_cat_v3
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = 42
N_SPLITS = 5
ORIGINAL_WEIGHT = 0.06
TE_SMOOTH = 20.0
LINEAGE_SMOOTH = 20.0
RARE = "__rare__"
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def balanced_error_metric(y_true, y_pred, sample_weight=None):
    yp = y_pred if y_pred.ndim == 2 else y_pred.reshape(-1, len(CLASSES))
    return 1.0 - balanced_accuracy_score(np.asarray(y_true).astype(int), np.argmax(yp, axis=1))


def make_xgb_params(seed, n_est, es):
    return {
        "objective": "multi:softprob", "num_class": len(CLASSES),
        "eval_metric": balanced_error_metric, "tree_method": "hist", "device": "cuda",
        "learning_rate": 0.012, "n_estimators": n_est, "early_stopping_rounds": es,
        "max_depth": 0, "max_leaves": 72, "grow_policy": "lossguide", "max_bin": 512,
        "min_child_weight": 10, "gamma": 0.20, "reg_alpha": 0.30, "reg_lambda": 4.0,
        "subsample": 0.82, "colsample_bytree": 0.74, "colsample_bylevel": 0.86,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }


def inv_freq_w(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    wpc = np.float32(len(y)) / (np.float32(len(CLASSES)) * np.maximum(counts, 1.0))
    return wpc[y].astype(np.float32)


def lineage_keys(alpha: np.ndarray, test_alphas: set) -> np.ndarray:
    """Exact-alpha key as a string categorical, but only for alphas present in the
    test set; everything else collapses to one RARE bucket → global prior under TE.
    Target-free (uses the test FEATURE, never its label)."""
    return np.array([str(a) if a in test_alphas else RARE for a in alpha]).reshape(-1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lineage-smooth", type=float, default=LINEAGE_SMOOTH)
    args = ap.parse_args()
    n_est = 400 if args.smoke else 7000
    es = 60 if args.smoke else 180
    lineage_smooth = args.lineage_smooth

    tr = pl.read_csv(DATA_DIR / "train.csv").sort("id").to_pandas()
    te = pl.read_csv(DATA_DIR / "test.csv").sort("id").to_pandas()
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.with_columns(pl.col("class").str.to_uppercase()).filter(
        pl.col("class").is_in(CLASSES)).to_pandas()
    y = tr["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    y_orig = orig["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    test_ids = te["id"].to_numpy()

    # ── alpha-lineage target-free pieces (aligned to the id-sorted frames) ──
    alpha_tr, alpha_te, alpha_or = (
        tr["alpha"].to_numpy(), te["alpha"].to_numpy(), orig["alpha"].to_numpy())
    test_alphas = set(alpha_te.tolist())
    cnt: Counter = Counter(alpha_tr.tolist())
    cnt.update(alpha_te.tolist())
    gs_tr = np.log1p([cnt[a] for a in alpha_tr]).astype(np.float32).reshape(-1, 1)
    gs_te = np.log1p([cnt[a] for a in alpha_te]).astype(np.float32).reshape(-1, 1)
    gs_or = np.log1p([cnt.get(a, 0) for a in alpha_or]).astype(np.float32).reshape(-1, 1)
    key_tr = lineage_keys(alpha_tr, test_alphas)
    key_te = lineage_keys(alpha_te, test_alphas)
    key_or = lineage_keys(alpha_or, test_alphas)
    n_lineage_keys = len(test_alphas)
    print(f"lineage: {len(test_alphas)} test-present alphas; "
          f"{(key_tr.ravel() != RARE).sum()} train rows keyed "
          f"({(key_tr.ravel() != RARE).mean():.1%})", flush=True)

    print("building v3 features...", flush=True)
    X, X_test, X_orig, cat_cols = build_features_cat_v3(tr, te, orig)
    num_cols = [c for c in X.columns if c not in cat_cols]
    Xc = X[cat_cols].astype("category")
    Xc_test = X_test[cat_cols].astype("category")
    Xc_orig = X_orig[cat_cols].astype("category")
    Xn = X[num_cols].to_numpy(np.float32)
    Xn_test = X_test[num_cols].to_numpy(np.float32)
    Xn_orig = X_orig[num_cols].to_numpy(np.float32)
    n_base = Xn.shape[1] + len(cat_cols)  # numeric + TE(cats) width before lineage
    lineage_names = [f"alpha_rate_{c}" for c in CLASSES] + ["alpha_log_groupsize"]
    print(f"X={X.shape}, {len(num_cols)} numeric + {len(cat_cols)} cats -> TE; "
          f"+{len(lineage_names)} lineage cols", flush=True)

    oof = np.zeros((len(X), 3), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")
    lineage_imp = np.zeros(len(lineage_names), dtype="float64")
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        print(f"===== fold {fold}/{len(folds)} =====", flush=True)
        enc = TargetEncoder(target_type="multiclass", smooth=TE_SMOOTH, cv=5, random_state=SEED)
        te_tr = enc.fit_transform(Xc.iloc[tri], y[tri]).astype(np.float32)
        te_va = enc.transform(Xc.iloc[vai]).astype(np.float32)
        te_te = enc.transform(Xc_test).astype(np.float32)
        te_or = enc.transform(Xc_orig).astype(np.float32)

        lenc = TargetEncoder(target_type="multiclass", smooth=lineage_smooth, cv=5,
                             random_state=SEED)
        ln_tr = lenc.fit_transform(key_tr[tri], y[tri]).astype(np.float32)
        ln_va = lenc.transform(key_tr[vai]).astype(np.float32)
        ln_te = lenc.transform(key_te).astype(np.float32)
        ln_or = lenc.transform(key_or).astype(np.float32)

        Xtr = np.hstack([Xn[tri], te_tr, ln_tr, gs_tr[tri]])
        Xva = np.hstack([Xn[vai], te_va, ln_va, gs_tr[vai]])
        Xte = np.hstack([Xn_test, te_te, ln_te, gs_te])
        Xor = np.hstack([Xn_orig, te_or, ln_or, gs_or])

        X_fit = np.vstack([Xtr, Xor])
        y_fit = np.concatenate([y[tri], y_orig])
        w = inv_freq_w(y_fit)
        w[len(tri):] *= np.float32(ORIGINAL_WEIGHT)

        model = XGBClassifier(**make_xgb_params(SEED + fold, n_est, es))
        model.fit(X_fit, y_fit, sample_weight=w, eval_set=[(Xva, y[vai])], verbose=False)
        oof[vai] = model.predict_proba(Xva).astype("float32")
        if not args.smoke:
            test_pred += model.predict_proba(Xte).astype("float32") / N_SPLITS
        imp = model.feature_importances_
        lineage_imp += imp[-len(lineage_names):]
        sc = balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))
        tot_imp = imp.sum() + 1e-12
        print(f"fold {fold} bal-acc {sc:.6f}  best_iter {model.best_iteration}", flush=True)
        print("  lineage importances (gain-share): " + ", ".join(
            f"{n}={i / tot_imp:.4f}" for n, i in zip(lineage_names, imp[-len(lineage_names):])),
            flush=True)
        del model, enc, lenc, X_fit, Xtr, Xva, Xte, Xor
        gc.collect()

    if args.smoke:
        print("smoke OK")
        return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"\nxgb_v3fe_lineage OOF argmax {argmax:.5f}  thresh {best:.5f}  "
          f"[xgb_v3fe 0.96711 / catboost_v3 0.96891]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    print(f"mean lineage importance over folds: " + ", ".join(
        f"{n}={v / len(folds):.1f}" for n, v in zip(lineage_names, lineage_imp)))
    np.save(RESULTS_DIR / "oof_xgb_v3fe_lineage.npy", oof)
    np.save(RESULTS_DIR / "test_xgb_v3fe_lineage.npy", test_pred)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / "threshold_weights_xgb_v3fe_lineage.json")
    save_cv_result(RESULTS_DIR, "xgb_v3fe_lineage", [], best, metric_name="balanced_acc")
    labels = [CLASSES[i] for i in np.argmax(test_pred * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, "class": labels}).to_csv(
        SUBMISSIONS_DIR / "xgb_v3fe_lineage.csv", index=False)
    print("Saved → xgb_v3fe_lineage")


if __name__ == "__main__":
    main()
