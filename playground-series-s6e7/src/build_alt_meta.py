"""Alternative meta-learners over the strong-5 m100 base OOFs (port of s6e6's).

Two metas the LR stack does NOT cover, both honest multi-seed outer 5-fold:

  ridgecal-gbdt : per-base fold-safe Ridge 3->3 calibration (inv-freq weighted,
      alpha=10) THEN a regularized LightGBM meta. The ridge rotation is what makes
      a tree meta non-redundant with the LR stacker: axis-aligned splits over
      per-base linearly-rotated logits can expose directions raw-logit splits miss,
      and a tree meta can do region-dependent model trust.
  nn : small sklearn MLP (64,32) over standardized logits, inv-freq oversampling.

Decision layer: robust_decision_weights (s6e7 convention). The meta factories and
ridge_calibrate are importable by probe_combiner_gate.py for the honest tournament.

Run: S6E7_REPAIR=1 python src/build_alt_meta.py --mode ridgecal-gbdt
     S6E7_REPAIR=1 python src/build_alt_meta.py --mode nn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_lr_stack import DEFAULT_KEYS
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR
from features import CLASSES, TARGET
from train_common import (
    MECHANISM_SHIFTED,
    N_CLASSES,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle.stacking import clipped_logit
from data_science_stuff.kaggle_utils import weighted_predict

SEEDS = [2024, 7, 13]


def inv_freq_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    return np.asarray(w[y])


def ridge_calibrate(
    per_base_tr: list[np.ndarray],
    eval_sets: list[list[np.ndarray]],
    y_tr: np.ndarray,
    alpha: float = 10.0,
) -> list[np.ndarray]:
    """Per-base fold-safe 3->3 ridge map (logits -> one-hot), fit on train split only.

    Returns [X_tr, X_eval1, X_eval2, ...] — one hstacked matrix per input set.
    """
    Y = np.eye(N_CLASSES)[y_tr]
    w = inv_freq_weights(y_tr)
    outs: list[list[np.ndarray]] = [[] for _ in range(1 + len(eval_sets))]
    for i, Btr in enumerate(per_base_tr):
        sc = StandardScaler().fit(Btr)
        r = Ridge(alpha=alpha).fit(sc.transform(Btr), Y, sample_weight=w)
        outs[0].append(r.predict(sc.transform(Btr)))
        for j, es in enumerate(eval_sets, 1):
            outs[j].append(r.predict(sc.transform(es[i])))
    return [np.hstack(o) for o in outs]


def gbdt_meta_factory(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def nn_meta_factory(seed: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        batch_size=4096,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=seed,
    )


def fit_gbdtmeta(
    blocks: list[np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """One ridgecal->LGBM meta fit; returns (probs_fit, probs_eval)."""
    Xtr, Xev = ridge_calibrate(
        [b[fit_idx] for b in blocks], [[b[eval_idx] for b in blocks]], y[fit_idx]
    )
    clf = gbdt_meta_factory(seed)
    clf.fit(Xtr, y[fit_idx])
    return clf.predict_proba(Xtr), clf.predict_proba(Xev)


def fit_nnmeta(
    blocks: list[np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """One oversampled MLP meta fit; returns (probs_fit, probs_eval)."""
    Z = np.hstack(blocks)
    sc = StandardScaler().fit(Z[fit_idx])
    Xtr, Xev = sc.transform(Z[fit_idx]), sc.transform(Z[eval_idx])
    rng = np.random.RandomState(seed)
    counts = np.bincount(y[fit_idx], minlength=N_CLASSES)
    idx = []
    for c in range(N_CLASSES):
        ci = np.where(y[fit_idx] == c)[0]
        idx.append(rng.choice(ci, size=counts.max(), replace=True))
    bal = np.concatenate(idx)
    rng.shuffle(bal)
    clf = nn_meta_factory(seed)
    clf.fit(Xtr[bal], y[fit_idx][bal])
    return clf.predict_proba(Xtr), clf.predict_proba(Xev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ridgecal-gbdt", "nn"], required=True)
    ap.add_argument("keys", nargs="*", default=None)
    args = ap.parse_args()
    keys = args.keys or DEFAULT_KEYS

    ds = load_dataset()
    y = ds.y
    c4 = (
        ds.train[[c for c in MECHANISM_SHIFTED if c in ds.train.columns]]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    blocks = [clipped_logit(np.load(RESULTS_DIR / f"oof_{k}.npy")) for k in keys]
    blocks_te = [clipped_logit(np.load(RESULTS_DIR / f"test_{k}.npy")) for k in keys]
    fit_one = fit_gbdtmeta if args.mode == "ridgecal-gbdt" else fit_nnmeta
    name = "gbdtmeta_r" if args.mode == "ridgecal-gbdt" else "nnmeta_r"
    print(f"[{name}] honest outer stack over {keys}")

    oof = np.zeros((len(y), N_CLASSES))
    test = np.zeros((len(ds.test_ids), N_CLASSES))
    n_test_fits = len(SEEDS) * 5
    _ = fit_one  # cell-level helpers are the gate's API; main inlines for one-fit test
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(np.zeros(len(y)), y):
            if args.mode == "ridgecal-gbdt":
                Xtr, Xva, Xte = ridge_calibrate(
                    [b[tri] for b in blocks],
                    [[b[vai] for b in blocks], blocks_te],
                    y[tri],
                )
                clf: LGBMClassifier | MLPClassifier = gbdt_meta_factory(seed)
                clf.fit(Xtr, y[tri])
            else:
                Z = np.hstack(blocks)
                sc = StandardScaler().fit(Z[tri])
                Xva = sc.transform(Z[vai])
                Xte = sc.transform(np.hstack(blocks_te))
                rng = np.random.RandomState(seed)
                counts = np.bincount(y[tri], minlength=N_CLASSES)
                idx = [
                    rng.choice(
                        np.where(y[tri] == c)[0], size=counts.max(), replace=True
                    )
                    for c in range(N_CLASSES)
                ]
                bal = np.concatenate(idx)
                rng.shuffle(bal)
                clf = nn_meta_factory(seed)
                clf.fit(sc.transform(Z[tri])[bal], y[tri][bal])
            oof[vai] += clf.predict_proba(Xva) / len(SEEDS)
            test += clf.predict_proba(Xte) / n_test_fits

    dw = robust_decision_weights(y, oof)
    pred = weighted_predict(oof, dw)
    print(
        f"[{name}] weighted={balanced_accuracy_score(y, pred):.4f} "
        f"complete4={balanced_accuracy_score(y[c4], pred[c4]):.4f} "
        f"miss4={balanced_accuracy_score(y[~c4], pred[~c4]):.4f}"
    )
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{name}.npy", test)
    with (RESULTS_DIR / f"decision_weights_{name}.json").open("w") as f:
        json.dump(dict(zip(CLASSES, dw.tolist())), f, indent=2)
    labels = np.array(CLASSES)[weighted_predict(test, dw)]
    pd.DataFrame({"id": ds.test_ids, TARGET: labels}).to_csv(
        SUBMISSIONS_DIR / f"{name}.csv", index=False
    )
    print(f"saved arrays + submissions/{name}.csv")


if __name__ == "__main__":
    main()
