"""Decorrelated NN base: RealMLP (logit-adjust) on the RICH v3 FE — new family-diversity (S6E6).

Our only strong NN (realmlp_deotte 0.96888) uses a MODEST FE; the FT-Transformer we tried earlier
went flat because it shared that same modest FE (correlated errors). This runs the SAME proven
from-scratch RealMLP (PBLD embeds, n_ens=8, EMA, balanced-softmax loss_prior_power=1.075) on the
rich CAT_v3 feature set instead — a different feature view -> genuinely decorrelated NN errors, the
thing the stacker rewards. Two NN-specific adaptations of the v3 FE:
  - only LOW-cardinality cats (<=100 levels) used as embeddings (the high-card hashed combos would
    blow up embedding tables; their signal is in the numerics anyway);
  - arcsinh-squash all numerics (the v3 set has heavy tails: fluxes ~1e12, band/redshift ~1e7 at
    low z) before RealMLP's robust-scale, so the NN doesn't blow up.

HP note: starts from realmlp_deotte's tuned CONFIG; --tune runs a small 1-fold sweep over the knobs
most sensitive to the wider input (lr, dropout, epochs, loss_prior_power) and uses the best.

Run:  python src/train_realmlp_v3fe.py [--smoke] [--tune] [--recover]
  --recover  Re-predict OOF/test from saved fold*.pth checkpoints (no retraining).
             Run this once after a crash to salvage completed folds, then re-run normally.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Reduce GPU memory fragmentation across folds (critical on 6GB card with multi-fold training)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from catboost_v3_features import CLASSES, build_features_cat_v3
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from realmlp_deotte import CONFIG, NumericalPreprocessor, RealMLP, RealMLP_TD_Classifier

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT_DIR = RESULTS_DIR / "_realmlp_v3fe_ckpt"
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
SEED, FOLDS = 42, 5
RUN = "realmlp_v3fe"
MAX_CARD = 100  # cats with <= this many levels become NN embeddings


def load_features():
    tr = pl.read_csv(DATA_DIR / "train.csv").sort("id").to_pandas()
    te = pl.read_csv(DATA_DIR / "test.csv").sort("id").to_pandas()
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.with_columns(pl.col("class").str.to_uppercase()).filter(
        pl.col("class").is_in(CLASSES)).to_pandas()
    y = tr["class"].map(CLASS_TO_INT).astype("int64").to_numpy()
    test_id = te["id"].to_numpy()
    X, X_test, _, cat_all = build_features_cat_v3(tr, te, orig)

    num_cols = [c for c in X.columns if c not in cat_all]
    cat_cols = sorted([c for c in cat_all if X[c].nunique() <= MAX_CARD])
    # arcsinh-squash heavy-tailed numerics for NN stability
    for c in num_cols:
        X[c] = np.arcsinh(X[c].to_numpy(np.float64)).astype("float32")
        X_test[c] = np.arcsinh(X_test[c].to_numpy(np.float64)).astype("float32")
    # re-key low-card cats to dense 0..k codes + category dtype (drop high-card)
    for c in cat_cols:
        cats = pd.Index(sorted(set(X[c]) | set(X_test[c])))
        m = {v: i for i, v in enumerate(cats)}
        X[c] = X[c].map(m).astype("int32").astype("category")
        X_test[c] = X_test[c].map(m).astype("int32").astype("category")
    keep = num_cols + cat_cols
    X, X_test = X[keep].copy(), X_test[keep].copy()
    print(f"X {X.shape}  numeric {len(num_cols)}  cat(<= {MAX_CARD}) {len(cat_cols)}", flush=True)
    return X, X_test, pd.Series(y), test_id, cat_cols


def _save_fold(fold: int, vai: np.ndarray, vp: np.ndarray, tp: np.ndarray) -> None:
    np.save(CKPT_DIR / f"fold{fold}_vai.npy", vai.astype(np.int32))
    np.save(CKPT_DIR / f"fold{fold}_oof.npy", vp)
    np.save(CKPT_DIR / f"fold{fold}_test.npy", tp)


def _load_fold(fold: int):
    """Return (vai, vp, tp) if all three npy files exist, else None."""
    vai_f = CKPT_DIR / f"fold{fold}_vai.npy"
    oof_f = CKPT_DIR / f"fold{fold}_oof.npy"
    tst_f = CKPT_DIR / f"fold{fold}_test.npy"
    if vai_f.exists() and oof_f.exists() and tst_f.exists():
        return np.load(vai_f), np.load(oof_f), np.load(tst_f)
    return None


def _recover_fold_from_pth(fold: int, tri: np.ndarray, vai: np.ndarray,
                           X: pd.DataFrame, X_test: pd.DataFrame,
                           cat_cols: list, cfg: dict) -> tuple | None:
    """Re-predict OOF/test from a saved fold*.pth without retraining.

    Reconstructs the preprocessor (deterministic: median/IQR fit on the same train split),
    loads the EMA state_dict, and runs inference on val + test.
    """
    ckpt_file = CKPT_DIR / f"fold{fold}.pth"
    if not ckpt_file.exists():
        print(f"  Fold {fold}: no .pth checkpoint found, must retrain.", flush=True)
        return None

    dev = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_tr = X.iloc[tri]
    X_val = X.iloc[vai]

    # Fit preprocessor identically to the original training run
    preprocessor = NumericalPreprocessor(cfg.get("tfms", ["median_center", "robust_scale"]))
    preprocessor.fit(X_tr[num_cols].values.astype(np.float32))

    # cat_dims: global max across train+test (same logic as RealMLP_TD_Classifier.fit)
    all_cat = np.concatenate(
        [X[cat_cols].values.astype(np.int64), X_test[cat_cols].values.astype(np.int64)], axis=0)
    cat_dims = (all_cat.max(axis=0) + 1).tolist()

    # Build model architecture and load weights
    rmlp = RealMLP(output_dim=3, cat_dims=cat_dims, n_numerical=len(num_cols), cfg=cfg).to(dev)
    state = torch.load(ckpt_file, map_location=dev, weights_only=True)
    rmlp.load_state_dict(state, strict=True)
    rmlp.eval()

    # Attach everything the predict_proba interface needs
    clf = RealMLP_TD_Classifier(**cfg)
    clf.preprocessor_ = preprocessor
    clf.num_col_names_ = num_cols
    clf.cat_col_names_ = cat_cols
    clf.cat_dims_ = cat_dims
    clf.model_ = rmlp
    clf._dev = dev
    clf.classes_ = np.arange(3)

    vp = clf.predict_proba(X_val).astype("float32")
    tp = clf.predict_proba(X_test).astype("float32")

    del rmlp, clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vp, tp


def fit_fold(cfg, X_tr, y_tr, X_val, y_val, cat_cols, X_tst, fold):
    torch.manual_seed(cfg["random_state"])
    np.random.seed(cfg["random_state"])
    model = RealMLP_TD_Classifier(**cfg)
    model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols,
              ckpt_path=str(CKPT_DIR / f"fold{fold}.pth"), X_test=X_tst)
    val_probs = model.best_val_probs_.astype("float32")
    test_probs = model.predict_proba(X_tst).astype("float32")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_probs, test_probs


def tune(X, y, cat_cols):
    """Small 1-fold sweep over the knobs most sensitive to the wider v3 input."""
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    tri, vai = next(iter(skf.split(X, y)))
    X_tr, X_val = X.iloc[tri].copy(), X.iloc[vai].copy()
    y_tr, y_val = y.iloc[tri], y.iloc[vai]
    grid = [
        {}, {"lr": 0.006}, {"lr": 0.014}, {"dropout": 0.10}, {"dropout": 0.0},
        {"epochs": 8}, {"loss_prior_power": 1.0}, {"loss_prior_power": 1.15},
        {"hidden_dims": [384, 384, 384]},
    ]
    best = (-1.0, None)
    for ov in grid:
        cfg = {**CONFIG, "random_state": SEED + 100, "verbosity": 0, **ov}
        vp, _ = fit_fold(cfg, X_tr, y_tr, X_val, y_val, cat_cols, X_val, 0)
        sc = balanced_accuracy_score(y_val, np.argmax(vp, axis=1))
        print(f"  tune {str(ov):42s} -> {sc:.5f}", flush=True)
        if sc > best[0]:
            best = (sc, ov)
    print(f"  best override {best[1]} ({best[0]:.5f})", flush=True)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--recover", action="store_true",
                    help="Re-predict OOF/test from saved fold*.pth checkpoints (no retraining).")
    args = ap.parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    X, X_test, y, test_id, cat_cols = load_features()

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    # --- recovery mode: re-predict from fold*.pth, save fold*_oof.npy, then exit ---
    if args.recover:
        print("=== RECOVERY MODE: re-predicting from saved checkpoints ===", flush=True)
        cfg_rec = {**CONFIG, "verbosity": 0}
        recovered = 0
        for fold, (tri, vai) in enumerate(folds, 1):
            if _load_fold(fold) is not None:
                print(f"  Fold {fold}: fold data already saved, skipping.", flush=True)
                continue
            result = _recover_fold_from_pth(fold, tri, vai, X, X_test, cat_cols, cfg_rec)
            if result is not None:
                vp, tp = result
                sc = balanced_accuracy_score(y.iloc[vai], np.argmax(vp, axis=1))
                print(f"  Fold {fold} recovered  balanced_acc={sc:.5f}", flush=True)
                _save_fold(fold, vai, vp, tp)
                recovered += 1
        print(f"Recovery done: {recovered} fold(s) saved.", flush=True)
        return

    override = {}
    if args.smoke:
        override = {"epochs": 2}
    elif args.tune:
        override = tune(X, y, cat_cols)

    oof = np.zeros((len(X), 3), dtype="float32")
    test_proba = np.zeros((len(X_test), 3), dtype="float32")
    fold_scores = []
    if args.smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        # --- resume: load from per-fold npy checkpoint if available ---
        saved = _load_fold(fold)
        if saved is not None and not args.smoke:
            vai_s, vp, tp = saved
            oof[vai_s] = vp
            test_proba += tp / FOLDS
            sc = float(balanced_accuracy_score(y.iloc[vai_s], np.argmax(vp, axis=1)))
            fold_scores.append(sc)
            print(f"  Fold {fold} RESUMED from checkpoint  balanced_acc: {sc:.5f}", flush=True)
            continue

        cfg = {**CONFIG, "random_state": SEED + fold * 100, **override}
        X_tr, X_val, X_tst = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        vp, tp = fit_fold(cfg, X_tr, y.iloc[tri], X_val, y.iloc[vai], cat_cols, X_tst, fold)
        oof[vai] = vp
        if not args.smoke:
            test_proba += tp / FOLDS
            _save_fold(fold, vai, vp, tp)   # persist immediately so crash doesn't lose this fold
        fold_scores.append(float(balanced_accuracy_score(y.iloc[vai], np.argmax(vp, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}", flush=True)
        del X_tr, X_val, X_tst

    if args.smoke:
        print("smoke OK")
        return

    y_np = y.to_numpy()
    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y_np, pred))
    rec = recall_score(y_np, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y_np)
    print(f"\n{RUN} OOF argmax {argmax:.5f}  thresh {best:.5f}  [realmlp_deotte 0.96888]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [CLASSES[i] for i in np.argmax(test_proba, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_id, "class": labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
