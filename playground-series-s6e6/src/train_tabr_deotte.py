"""TabR (pytabkit RealTabR_D) on cdeotte's 240-feature recipe — a genuinely NEW model family.

The stack only ever moved on a DIFFERENT model FAMILY (RealMLP +0.0019; CatBoost +0.0003 when we
had none). Every same-family base (GBDT/RealMLP/TabM on deotte-FE) is flat. TabR (Gorishniy et al.
2023) is the one untried inductive bias: a retrieval-augmented NN whose hidden layers are a learned
k-NN attention over a CONTEXT of candidate rows — not axis-aligned tree splits, not a plain
MLP/attention on features. If the low-z GALAXY/STAR confusion region has manifold structure that a
learned metric-space retrieval can exploit differently, TabR's errors will be decorrelated from the
cluster. RealTabR_D = TabR with RealMLP-style improvements (scaling layer, NTP linear, num
embeddings) — the variant closest to our proven realmlp_deotte lever.

Reuses the xgb_deotte pipeline (build_feature_matrix + per-fold fold-safe qbin TE); native cats.
id-sorted I/O. Per-fold checkpoints (machine has documented random reboots — a 5-fold run must
resume, not restart). val_metric_name="1-balanced_accuracy" for leakage-safe internal early stop.

⚠️ 6GB GPU + TabR retrieval over 577k candidates = OOM risk. memory_efficient=True +
candidate_encoding_batch_size cap the peak. SMOKE FIRST: TABR_MAX_FOLD=1 runs fold 1 only.
GATE: standalone ~0.965+ AND decorrelated (watch STAR recall + eval_stack_delta). A flat result =
THIS family flat (a real finding — the new-family lever finally exhausted), not an execution miss.

Run:  TABR_MAX_FOLD=1 python -u src/train_tabr_deotte.py    # smoke (1 fold, no save)
      python -u src/train_tabr_deotte.py                    # full 5-fold
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealTabR_D_Classifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from deotte_features import (
    CLASS_TO_INT, CLASSES, ID_COL, INT_TO_CLASS, TARGET, TOP_FEATURES,
    build_feature_matrix,
)
from postprocess import optimize_thresholds, save_threshold_weights
from train_xgb_deotte import add_fold_safe_te, sorted_factorize, te_sources

warnings.filterwarnings("ignore")
from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
CKPT = RESULTS_DIR / "_tabr_deotte_ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED, N_SPLITS = 42, 5
# RealTabR_D does NOT support compile_model/allow_amp (TabrConstructorMixin, different from TabM).
# Speed levers: larger batches + freeze_contexts_after_n_epochs (stops re-encoding 577k candidates
# every step after N warm-up epochs — the main compute bottleneck in TabR retrieval).
BATCH_SIZE = 512           # default "auto" ~256; 512 safe at fp32 with memory_efficient=True
EVAL_BATCH_SIZE = 1024     # default 4096; must be small — val allocates eval_bs×96×265 dims at once
CAND_ENC_BS = 2048         # candidate_encoding_batch_size — keep conservative (primary OOM knob)
FREEZE_CTX_EPOCHS = 3      # freeze candidate encodings after 3 warm-up epochs
RUN = "tabr_deotte"
MAX_FOLD = int(os.environ.get("TABR_MAX_FOLD", N_SPLITS))  # =1 → fold 1 only, no final save


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"Device {DEVICE}  bs={BATCH_SIZE} eval_bs={EVAL_BATCH_SIZE} cand_enc_bs={CAND_ENC_BS}")
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
            # pytabkit's TabR interface chokes on cat_col_names (empty-array ordinal-encode
            # crash); factorize the qbin cats to int codes and feed ALL features as numeric.
            # The deotte FE already encodes categorical signal numerically via the TE columns,
            # and TabR's lever is the retrieval mechanism, not native cat embedding.
            for c in cats:
                X_tr[c], X_va[c], X_te[c] = sorted_factorize(X_tr[c], X_va[c], X_te[c])
            # TabR supports only 'class_error' (val_accuracy) or 'cross_entropy'. Raw accuracy
            # biases early-stop toward majority GALAXY → use cross_entropy (class-agnostic, the
            # safer proxy for our balanced-acc objective on a 65/20/14 imbalance). NOTE: like TabM,
            # RealTabR has no class-weight/logit-adjust hook — STAR recall rides on the FE's TE cols.
            model = RealTabR_D_Classifier(
                device=DEVICE, random_state=SEED + fold,
                val_metric_name="cross_entropy",
                batch_size=BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
                candidate_encoding_batch_size=CAND_ENC_BS,
                memory_efficient=True,
                freeze_contexts_after_n_epochs=FREEZE_CTX_EPOCHS,
                verbosity=1,
            )
            model.fit(X_tr[feats], y[tri])
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
            print(f"\n[SMOKE] stopped after fold {fold} (TABR_MAX_FOLD={MAX_FOLD}); no final save.")
            print(f"  GO/NO-GO: standalone ~0.965+ AND STAR recall healthy → finish 5 folds + eval_stack_delta.")
            return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [realmlp_deotte 0.96888; new FAMILY = the bet]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INT_TO_CLASS[i] for i in np.argmax(test_proba * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels, id_col=ID_COL)
    for f in CKPT.glob("f*.npy"):
        f.unlink()
    CKPT.rmdir()
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
