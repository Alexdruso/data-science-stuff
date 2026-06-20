"""Confident-learning variant of catboost_v3 (Plan C, S6E6).

Hypothesis: the synthetic labels carry noise; the rows where our STRONG stack (lrstack OOF) gives
<tau probability to the row's OWN assigned label are mislabel candidates. Refitting the strongest
base (catboost_v3) with those rows DROPPED FROM TRAINING — but still PREDICTED in OOF — de-noises
the decision boundary and may yield a base that's decorrelated enough for the stacker to reward.
This changes the training SIGNAL (vs same-label refits which were flat), so it's a genuinely
different axis from xgb_v3fe etc.

Leakage: the drop flag uses lrstack OOF, which is out-of-fold per row (the row's own label never
fit its own lrstack pred), and we only drop from the fold's TRAIN split — the val fold is scored in
full → OOF stays complete and aligned (CLAUDE.md Rule 4). Original SDSS17 rows are never dropped.

Caveat (advisor): the dropped <1-2% are a MIX of label noise (dropping helps) and hard-but-correct
low-z STAR/GAL overlap (dropping removes boundary-defining rows → hurts). Hence the tau sweep + a
stack-delta gate, not a single cut.

GATE: stack delta via eval_stack_delta.py (swap catboost_v3 -> catboost_v3_cl) over 0.970481.

Run:  python src/train_catboost_v3_cl.py [--smoke] [--drop-thresh 0.02]
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from catboost_v3_features import CLASSES, build_features_cat_v3
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from train_catboost_v3 import cat_params, load_sorted, predict_batched

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = 42
N_SPLITS = 5
ORIGINAL_WEIGHT = 0.06
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop-thresh", type=float, default=0.02)
    args = ap.parse_args()
    iterations = 300 if args.smoke else 5000
    es = 60 if args.smoke else 260
    tau = args.drop_thresh
    run = "catboost_v3_cl"

    tr, te, orig = load_sorted()
    y = tr["class"].map(CLASS_TO_INT).astype("int8").to_numpy()
    y_orig = orig["class"].map(CLASS_TO_INT).astype("int8").to_numpy()
    test_ids = te["id"].to_numpy()

    # noisy-row flag from the strong stack OOF (out-of-fold per row, id-sorted == X order)
    oof_lr = np.load(RESULTS_DIR / "oof_lrstack.npy")
    p_true = oof_lr[np.arange(len(y)), y]
    noisy = p_true < tau
    print(f"CL drop tau={tau}: flag {noisy.sum()} rows ({noisy.mean():.2%}) as mislabel candidates",
          flush=True)

    print(f"building features...", flush=True)
    X, X_test, X_orig, cat_cols = build_features_cat_v3(tr, te, orig)
    print(f"X={X.shape}, cats={len(cat_cols)}", flush=True)

    oof = np.zeros((len(X), 3), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        tri_clean = tri[~noisy[tri]]  # drop flagged rows from TRAIN only; val stays full
        print(f"===== fold {fold}/{len(folds)} =====  train {len(tri)} -> {len(tri_clean)} "
              f"(dropped {len(tri) - len(tri_clean)})", flush=True)
        X_fit = pd.concat([X.iloc[tri_clean], X_orig], axis=0, ignore_index=True)
        y_fit = np.concatenate([y[tri_clean], y_orig]).astype("int8")
        w = np.ones(len(y_fit), dtype="float32")
        w[len(tri_clean):] = np.float32(ORIGINAL_WEIGHT)
        train_pool = Pool(X_fit, y_fit, cat_features=cat_cols, weight=w)
        valid_pool = Pool(X.iloc[vai], y[vai], cat_features=cat_cols)

        model = CatBoostClassifier(**cat_params(SEED + fold, iterations, es))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        oof[vai] = model.predict_proba(valid_pool).astype("float32")
        if not args.smoke:
            test_pred += predict_batched(model, X_test, cat_cols) / N_SPLITS
        sc = balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))
        print(f"fold {fold} bal-acc {sc:.6f}  best_iter {model.get_best_iteration()}", flush=True)
        del model, train_pool, valid_pool, X_fit, y_fit, w
        gc.collect()

    if args.smoke:
        print("smoke OK")
        return

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"\n{run} (tau={tau}) OOF argmax {argmax:.5f}  thresh {best:.5f}  "
          f"[catboost_v3 0.96891]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    np.save(RESULTS_DIR / f"oof_{run}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run}.npy", test_pred)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{run}.json")
    save_cv_result(RESULTS_DIR, run, [], best, metric_name="balanced_acc")
    labels = [CLASSES[i] for i in np.argmax(test_pred * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, "class": labels}).to_csv(SUBMISSIONS_DIR / f"{run}.csv", index=False)
    print(f"Saved → {run}")


if __name__ == "__main__":
    main()
