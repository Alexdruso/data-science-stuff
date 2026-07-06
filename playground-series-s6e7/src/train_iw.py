"""Adversarial importance-weighting probe for PS S6E7 — train TOWARD the test distribution.

Today's core finding is a statement about the covariate shift: every diverse base fixes ~72% of
its errors in NON-test-like rows, so its gains don't transfer. The never-actually-run counter is
to reweight training so the model optimizes the test-like region directly. Importance weight for
covariate shift = p_test(x)/p_train(x); we already have the adversarial P(test|x) per train row
(results/adv_scores_train.npy), so w_i = odds = P(test|x_i)/(1-P(test|x_i)), clipped to [0.2,5] to
avoid a few rows dominating. Combined with the balanced class weight (kept — it's the imbalance
lever), passed as a single sample_weight. Zero leakage: only reweights real labelled train rows.

Gate lgbm_iw with diag_mlp_transfer.py: it must CLEARLY lift adv-weighted / test-like OOF (that is
literally what it optimizes for) — if even that is flat, shift-targeting is measured-dead here,
consistent with CV≈LB in absolute terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
from baseline import load_params
from features import CLASSES
from train_common import N_CLASSES, RESULTS_DIR, Dataset, bagged_cv, finalize, load_dataset

ADV_CACHE = RESULTS_DIR / "adv_scores_train.npy"
CLIP = (0.2, 5.0)


def importance_weights(y: np.ndarray) -> np.ndarray:
    """balanced class weight × clipped test/train odds from the adversarial P(test|x)."""
    adv = np.load(ADV_CACHE)
    assert adv.shape == (len(y),), f"adv cache {adv.shape} != {(len(y),)}"
    odds = np.clip(adv / (1.0 - adv + 1e-9), *CLIP)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)
    return (cw[y] * odds).astype(np.float64)


def run(ds: Dataset) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    sw_all = importance_weights(ds.y)
    print(f"importance weights: mean {sw_all.mean():.3f} range [{sw_all.min():.2f},{sw_all.max():.2f}]")
    params = load_params()
    params.pop("class_weight", None)  # imbalance folded into sample_weight already

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001,ANN202
        model = LGBMClassifier(**{**params, "class_weight": None, "random_state": seed})
        model.fit(
            X.iloc[tr_idx],
            ds.y[tr_idx],
            sample_weight=sw_all[tr_idx],
            eval_set=[(X.iloc[val_idx], ds.y[val_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        return model.predict_proba(X.iloc[val_idx]), model.predict_proba(X_test)

    return bagged_cv(ds, fit_fold, seeds=[42])


def main() -> None:
    ds = load_dataset()
    print(f"Train: {ds.train.shape}   Test: {ds.test.shape}   classes: {list(CLASSES)}")
    oof, test, fold_scores = run(ds)
    finalize("lgbm_iw", ds, oof, test, fold_scores)


if __name__ == "__main__":
    main()
