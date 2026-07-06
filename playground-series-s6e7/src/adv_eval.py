"""Adversarial-validation-weighted OOF evaluation for PS S6E7.

Problem: plain OOF balanced accuracy is saturated (~0.949) and CANNOT rank configs
separated by <0.001 — the adv-AUC-0.65 covariate shift means OOF ↔ LB disagree at that
scale (bagged+MLP ensemble had higher OOF 0.9495 but LOWER LB 0.94910 than the non-bagged
3-GBDT champion 0.94970). This tool builds an offline metric that upweights train rows most
resembling the test distribution, to (a) check whether it ranks configs closer to LB and
(b) see which models transfer to the test-like region (is the MLP the LB regression culprit?).

Read-only over existing results/oof_*.npy — no training of the competition models, no submit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

GBDTS = ["lgbm", "xgboost", "catboost"]


def adversarial_scores(ds: object) -> NDArray[np.float64]:
    """OOF P(row is from test) for each TRAIN row, via a 5-fold train-vs-test classifier."""
    tr, te = ds.train.copy(), ds.test.copy()  # type: ignore[attr-defined]
    cat_cols = ds.cat_cols  # type: ignore[attr-defined]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    import pandas as pd

    X = pd.concat([tr, te], ignore_index=True)
    for c in cat_cols:
        X[c] = X[c].astype("category")
    is_test = np.concatenate([np.zeros(len(tr)), np.ones(len(te))]).astype(np.int64)

    oof = np.zeros(len(X))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X, is_test):
        m = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        m.fit(X.iloc[tr_idx], is_test[tr_idx])
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
    print(f"adversarial AUC: {roc_auc_score(is_test, oof):.4f}")
    return oof[: len(tr)]  # train rows only


def precorrected_blend(
    oof: dict[str, NDArray], models: list[str]
) -> NDArray[np.float64]:
    """Equal-weight blend of each model's decision-corrected probs (deployed surface)."""
    import json

    parts = []
    for m in models:
        with (RESULTS_DIR / f"decision_weights_{m}.json").open() as f:
            wd = json.load(f)
        from features import CLASSES

        w = np.array([wd[c] for c in CLASSES])
        p = oof[m] * w
        parts.append(p / p.sum(axis=1, keepdims=True))
    return np.mean(parts, axis=0)


def weighted_bacc(
    y: NDArray[np.int64], pred: NDArray[np.int64], sw: NDArray[np.float64]
) -> float:
    """Balanced accuracy with per-sample weights (weighted mean of per-class recalls)."""
    recalls = []
    for c in range(3):
        mask = y == c
        recalls.append(float(np.average(pred[mask] == c, weights=sw[mask])))
    return float(np.mean(recalls))


def main() -> None:
    ds = load_dataset()
    y = ds.y
    adv = adversarial_scores(ds)
    q = np.quantile(adv, 0.70)
    testlike = adv >= q  # top 30% most test-like train rows
    print(f"test-like subset: {testlike.sum()} rows (adv>={q:.3f})\n")

    oof = {m: np.load(RESULTS_DIR / f"oof_{m}.npy") for m in GBDTS + ["mlp"]}

    configs = {"3-GBDT": GBDTS, "3-GBDT+MLP": GBDTS + ["mlp"]}
    print(f"{'config':<12} {'plain_bacc':>10} {'testlike_bacc':>14} {'advwt_bacc':>11}")
    for name, models in configs.items():
        blend = precorrected_blend(oof, models)
        dw = robust_decision_weights(y, blend)
        pred = np.argmax(blend * dw, axis=1)
        plain = balanced_accuracy_score(y, pred)
        tl = balanced_accuracy_score(y[testlike], pred[testlike])
        aw = weighted_bacc(y, pred, adv)  # weight every row by its test-likeness
        print(f"{name:<12} {plain:>10.4f} {tl:>14.4f} {aw:>11.4f}")

    # Per-model on the test-like subset — does the MLP hold up where the GBDTs do?
    print("\nper-model balanced acc (decision-weighted argmax):")
    print(f"  {'model':<10} {'plain':>8} {'testlike':>9}")
    for m in GBDTS + ["mlp"]:
        dw = robust_decision_weights(y, oof[m])
        pred = np.argmax(oof[m] * dw, axis=1)
        print(
            f"  {m:<10} {balanced_accuracy_score(y, pred):>8.4f} "
            f"{balanced_accuracy_score(y[testlike], pred[testlike]):>9.4f}"
        )


if __name__ == "__main__":
    main()
