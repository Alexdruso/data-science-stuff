"""Alternative meta-learners over the 7 strong base OOFs — cheap reshuffle checks (S6E6).

Two metas the LR-stack (0.96977) / GBDT-stack (0.96981) incumbents do NOT cover:

  --mode ridgecal-gbdt : per-base 3->3 RIDGE calibration (fold-safe), THEN the regularized
      LightGBM meta. This is the one place ridge is NOT a linear-o-linear no-op vs the LR
      stacker: a tree is axis-aligned, so a per-base linear *rotation* of the 3 logits before
      the tree sees them can expose a discriminating direction the raw-logit splits miss.

  --mode nn : a small MLP meta over the 21 base logits (different meta FAMILY than LR/GBDT).
      Inv-freq oversampling of the train fold (leakage-safe) emulates class_weight="balanced".

Both run honest outer 5-fold x multi-seed OOF stacking on the (already out-of-fold) base logits.
GATE: OOF argmax must beat GBDT-stack 0.96981 to be a real lever.

Run:  python src/build_alt_meta.py --mode ridgecal-gbdt
      python src/build_alt_meta.py --mode nn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from lgbm_device import get_lgbm_device
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte", "catboost_v3", "xgb_v3fe",
          "chain_cascade", "chain_cascade_xgb"]
SEEDS = [2024, 7, 13]
GATE = 0.97055  # lrstack thresh-tuned best (2026-06-09, 10 bases incl chain_cascade)
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def inv_freq_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    w = len(y) / (3.0 * np.maximum(counts, 1.0))
    return w[y]


def ridge_calibrate(per_base_tr, per_base_va, per_base_te, y_tr, alpha=10.0):
    """Per-base fold-safe 3->3 ridge map (logits -> one-hot), fit on train split only."""
    Y = np.eye(3)[y_tr]
    w = inv_freq_weights(y_tr)
    cal_tr, cal_va, cal_te = [], [], []
    for Btr, Bva, Bte in zip(per_base_tr, per_base_va, per_base_te):
        sc = StandardScaler().fit(Btr)
        r = Ridge(alpha=alpha).fit(sc.transform(Btr), Y, sample_weight=w)
        cal_tr.append(r.predict(sc.transform(Btr)))
        cal_va.append(r.predict(sc.transform(Bva)))
        cal_te.append(r.predict(sc.transform(Bte)))
    return np.hstack(cal_tr), np.hstack(cal_va), np.hstack(cal_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ridgecal-gbdt", "nn"], required=True)
    args = ap.parse_args()
    dev_type, n_jobs = get_lgbm_device()

    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    # per-base logit blocks (each n x 3), kept separate for per-base calibration
    blocks_tr, blocks_te, used = [], [], []
    for m in MODELS:
        fo, ft = RESULTS_DIR / f"oof_{m}.npy", RESULTS_DIR / f"test_{m}.npy"
        if fo.exists() and ft.exists():
            blocks_tr.append(logit(np.load(fo)))
            blocks_te.append(logit(np.load(ft)))
            used.append(m)
    print(f"[{args.mode}] meta over {len(used)} bases: {used}  (GATE {GATE})")
    Ztr_all = np.hstack(blocks_tr)
    Zte_all = np.hstack(blocks_te)

    oof = np.zeros((len(y), 3))
    test = np.zeros((len(test_ids), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Ztr_all, y):
            if args.mode == "ridgecal-gbdt":
                per_tr = [b[tri] for b in blocks_tr]
                per_va = [b[vai] for b in blocks_tr]
                Xtr, Xva, Xte = ridge_calibrate(per_tr, per_va, blocks_te, y[tri])
                params = dict(
                    objective="multiclass", num_class=3, class_weight="balanced",
                    n_estimators=300, learning_rate=0.02, num_leaves=15,
                    min_child_samples=200, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                    verbose=-1, n_jobs=n_jobs, device_type=dev_type,
                )
                clf = LGBMClassifier(**params)
                clf.fit(Xtr, y[tri])
                oof[vai] += clf.predict_proba(Xva) / len(SEEDS)
                test += clf.predict_proba(Xte) / (len(SEEDS) * 5)
            else:  # nn meta over standardized logits, inv-freq oversampling = balanced
                sc = StandardScaler().fit(Ztr_all[tri])
                Xtr = sc.transform(Ztr_all[tri])
                Xva, Xte = sc.transform(Ztr_all[vai]), sc.transform(Zte_all)
                rng = np.random.RandomState(seed)
                counts = np.bincount(y[tri], minlength=3)
                target = counts.max()
                idx = []
                for c in range(3):
                    ci = np.where(y[tri] == c)[0]
                    idx.append(rng.choice(ci, size=target, replace=True))
                bal = np.concatenate(idx)
                rng.shuffle(bal)
                clf = MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                    batch_size=4096, learning_rate_init=1e-3, max_iter=60,
                    early_stopping=True, n_iter_no_change=6, random_state=seed,
                )
                clf.fit(Xtr[bal], y[tri][bal])
                oof[vai] += clf.predict_proba(Xva) / len(SEEDS)
                test += clf.predict_proba(Xte) / (len(SEEDS) * 5)

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    mark = "  <== BEAT GATE" if argmax > GATE else ""
    print(f"[{args.mode}] OOF argmax {argmax:.5f}  thresh {best:.5f}{mark}")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  [GBDT-stack GAL .954/QSO .975/STAR .969]")

    run = f"altmeta_{args.mode}"
    save_cv_result(RESULTS_DIR, run, [], best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{run}.npy", oof)
    np.save(RESULTS_DIR / f"test_{run}.npy", test)
    if argmax > GATE:
        save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{run}.json")
        labels = le.inverse_transform(np.argmax(test * tw, axis=1))
        write_submission(SUBMISSIONS_DIR, f"{run}.csv", test_ids, TARGET, labels)
        print(f"Saved → {run} (beat gate)")
    else:
        print("Did not beat gate — oof/test saved for the record, no submission.")


if __name__ == "__main__":
    main()
