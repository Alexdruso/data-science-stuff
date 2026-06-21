"""TabM (pytabkit) on cdeotte's 240-feature recipe — a new diverse family (S6E6 campaign).

The stack only moves on DECORRELATED bases (RealMLP +0.0019; same-FE CatBoost +0). TabM
(Gorishniy 2024, parameter-efficient MLP ensembling) is a different architecture than our
RealMLP/XGB. Earlier TabM hit the STAR wall (0.909) — but that was on RAW features; here it gets
the rich recipe whose qbin TARGET-ENCODING features hand the model per-class signal directly, the
exact thing the raw-feature NNs lacked. Reuses the xgb_deotte feature pipeline (build_feature_matrix
+ per-fold fold-safe qbin TE); qbin cats passed NATIVELY (pytabkit embeds them). Internal val split
for early stopping (val_metric_name="1-balanced_accuracy") → leakage-safe. id-sorted I/O.

GATE: standalone ~0.965+ AND decorrelated (watch STAR recall + stack contribution).

CONFIG NOTE (2026-06-13): the first run used the stock TabM_D preset, which ships
num_emb_type='none' (piecewise-linear numerical embeddings OFF). Result: argmax 0.9564 /
threshold-tuned 0.9657, STAR recall only 0.9281 — UNDER-configured, not a family limit
(kirill0212's TabM reaches 0.96862 via this same API; realmlp_deotte's strength came from its
PBLD embeddings). Fix = num_emb_type='pwl' (the piecewise-linear embedding analog). A flat result
here = THIS config flat, NOT "NN-family limit"; do not pivot on that framing.

Run:  python src/train_tabm_deotte.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES, build_feature_matrix, cat_key
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import add_fold_safe_te, te_sources

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT = RESULTS_DIR / "_tabm_deotte_ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED, N_SPLITS = 42, 5
# 256 (fp32) → 512 safe with AMP (fp16 activations halve peak memory); 2048 still OOMs
BATCH_SIZE, LR = 512, 0.002
RUN = "tabm_deotte"
MAX_FOLD = int(os.environ.get("TABM_MAX_FOLD", N_SPLITS))  # gate: TABM_MAX_FOLD=1 runs fold 1 only, no final save


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    print(f"Device {DEVICE}  bs={BATCH_SIZE} lr={LR}")
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
    oof = np.zeros((len(X), len(CLASSES)))
    test_proba = np.zeros((len(X_test), len(CLASSES)))
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        vck, tck = CKPT / f"f{fold}_val.npy", CKPT / f"f{fold}_test.npy"
        if vck.exists() and tck.exists():
            val_pred, test_pred = np.load(vck), np.load(tck)
            print(f"  Fold {fold}: loaded checkpoint")
        else:
            X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
            X_tr, X_va, X_te = add_fold_safe_te(X_tr, y[tri], X_va, X_te, TE_COLS)
            feats = [f for f in TOP_FEATURES if f in X_tr.columns]
            cats = [c for c in MODEL_CAT_COLS if c in feats]
            for df in (X_tr, X_va, X_te):
                for c in cats:
                    df[c] = cat_key(df[c])
            model = TabM_D_Classifier(device=DEVICE, random_state=SEED + fold,
                                      val_metric_name="1-balanced_accuracy",
                                      num_emb_type="pwl",  # piecewise-linear num embeddings; the STAR lever
                                      batch_size=BATCH_SIZE, lr=LR, verbosity=1,
                                      compile_model=True,   # Inductor JIT (~10-30% speedup)
                                      allow_amp=True,       # fp16 Tensor Cores on RTX 2060 (~2-3x speedup)
                                      )
            model.fit(X_tr[feats], y[tri], cat_col_names=cats)
            assert list(model.classes_) == list(range(len(CLASSES))), model.classes_
            val_pred = model.predict_proba(X_va[feats])
            test_pred = model.predict_proba(X_te[feats])
            np.save(vck, val_pred); np.save(tck, test_pred)
            del model, X_tr, X_va, X_te
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        oof[vai] = val_pred
        test_proba += test_pred / N_SPLITS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_pred, axis=1))))
        rec_f = recall_score(y[vai], np.argmax(val_pred, axis=1), average=None, labels=[0, 1, 2])
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}  recall(G,Q,S) {rec_f.round(4)}")
        if fold >= MAX_FOLD and MAX_FOLD < N_SPLITS:
            print(f"\n[GATE] stopped after fold {fold} (TABM_MAX_FOLD={MAX_FOLD}); no final OOF/submission saved.")
            print("  Checkpoints kept for resume. STAR recall vs prev-config 0.9281 is the go/no-go signal.")
            return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [raw-feat TabM STAR-walled 0.909; realmlp_deotte 0.96888]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}  [watch STAR vs the 0.909 wall]")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    for f in CKPT.glob("f*.npy"):
        f.unlink()
    CKPT.rmdir()
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
