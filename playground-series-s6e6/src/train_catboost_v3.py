"""Port of cdeotte's CAT_v3 CatBoost (his CV 0.96897) — close our biggest base gap (0.96636).

Faithful to his recipe: the big categorical-heavy FE (catboost_v3_features), original SDSS17 rows
appended per fold at weight 0.06, and his exact CatBoost params (depth 8, lr 0.042, l2 8,
random_strength 1.2, Bayesian bagging 0.2, max_ctr_complexity 3, class_weights [1,3.25,5],
eval_metric TotalF1:Macro, 5000 iters, es 260). Frames are id-sorted so oof/test align with our
stack convention (CLAUDE.md Rule 4). Single-GPU here (he used 2x T4) -> devices='0'.

GATE: OOF argmax vs his 0.96897 and our catboost_deotte 0.96636. If it clears ~0.968, add to the
build_lr_stack / build_gbdt_stack MODELS list.

Run:  python src/train_catboost_v3.py [--smoke]
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

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
SEED = 42
N_SPLITS = 5
ORIGINAL_WEIGHT = 0.06
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}  # GALAXY0 QSO1 STAR2 (== our LabelEncoder)
PREDICT_BATCH = 80_000


def cat_params(seed, iterations, es):
    return {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": iterations,
        "depth": 8,
        "learning_rate": 0.042,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.2,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.2,
        "one_hot_max_size": 16,
        "max_ctr_complexity": 3,
        "class_weights": [1.0, 3.25, 5.0],
        "border_count": 254,
        "random_seed": seed,
        "early_stopping_rounds": es,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.85,
        "gpu_cat_features_storage": "CpuPinnedMemory",
        "thread_count": 4,
        "allow_writing_files": False,
        "verbose": 250,
    }


def predict_batched(model, X_data, cat_cols):
    parts = []
    for s in range(0, len(X_data), PREDICT_BATCH):
        pool = Pool(X_data.iloc[s:s + PREDICT_BATCH], cat_features=cat_cols)
        parts.append(model.predict_proba(pool).astype("float32"))
        del pool
        gc.collect()
    return np.vstack(parts).astype("float32")


def load_sorted():
    tr = pl.read_csv(DATA_DIR / "train.csv").sort("id").to_pandas()
    te = pl.read_csv(DATA_DIR / "test.csv").sort("id").to_pandas()
    orig = pl.read_csv(DATA_DIR / "star_classification.csv", infer_schema_length=None)
    orig = orig.with_columns(pl.col("class").str.to_uppercase()).filter(
        pl.col("class").is_in(CLASSES)).to_pandas()
    return tr, te, orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    iterations = 300 if args.smoke else 5000
    es = 60 if args.smoke else 260

    tr, te, orig = load_sorted()
    y = tr["class"].map(CLASS_TO_INT).astype("int8").to_numpy()
    y_orig = orig["class"].map(CLASS_TO_INT).astype("int8").to_numpy()
    test_ids = te["id"].to_numpy()

    print(f"building features (train {len(tr)} / test {len(te)} / orig {len(orig)})...", flush=True)
    X, X_test, X_orig, cat_cols = build_features_cat_v3(tr, te, orig)
    print(f"X={X.shape}, X_test={X_test.shape}, X_orig={X_orig.shape}, cats={len(cat_cols)}", flush=True)

    oof = np.zeros((len(X), 3), dtype="float32")
    test_pred = np.zeros((len(X_test), 3), dtype="float32")
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    if args.smoke:
        folds = folds[:1]

    for fold, (tri, vai) in enumerate(folds, 1):
        print(f"===== fold {fold}/{len(folds)} =====", flush=True)
        X_fit = pd.concat([X.iloc[tri], X_orig], axis=0, ignore_index=True)
        y_fit = np.concatenate([y[tri], y_orig]).astype("int8")
        w = np.ones(len(y_fit), dtype="float32")
        w[len(tri):] = np.float32(ORIGINAL_WEIGHT)
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
    print(f"\ncatboost_v3 OOF argmax {argmax:.5f}  thresh {best:.5f}  "
          f"[Deotte 0.96897 / our catboost_deotte 0.96636]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")

    np.save(RESULTS_DIR / "oof_catboost_v3.npy", oof)
    np.save(RESULTS_DIR / "test_catboost_v3.npy", test_pred)
    save_threshold_weights(tw, CLASSES, RESULTS_DIR / "threshold_weights_catboost_v3.json")
    save_cv_result(RESULTS_DIR, "catboost_v3", [], best, metric_name="balanced_acc")
    labels = [CLASSES[i] for i in np.argmax(test_pred * tw, axis=1)]
    write_submission(SUBMISSIONS_DIR, "catboost_v3.csv", test_ids, "class", labels)
    print("Saved → catboost_v3 (oof/test/submission)")


if __name__ == "__main__":
    main()
