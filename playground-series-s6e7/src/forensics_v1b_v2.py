"""Test-side disagreement forensics: champion (v1b) vs v2 (bag+MLP) predictions.

We know v1b LB .94970 > v2 LB .94910, i.e. the net effect of the rows where they
disagree is negative. This characterizes those flipped test rows (class transitions,
adversarial test-likeness, key-driver missingness) to localize what v2 changed.

Also saves the reconstructed champion test labels → results/champion_test_labels.npy
(the scored v1b CSV was overwritten before being preserved).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict
from features import CLASSES
from lb_anchor import KEY_DRIVERS, correct
from train_common import DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR, load_dataset

GBDTS = ["lgbm", "xgboost", "catboost"]


def champion_test_labels(y: NDArray) -> NDArray:
    """Reproduce the v1b recipe end-to-end on the _v1 arrays and return test label idx."""
    oof = {m: np.load(RESULTS_DIR / f"oof_{m}_v1.npy") for m in GBDTS}
    test = {m: np.load(RESULTS_DIR / f"test_{m}_v1.npy") for m in GBDTS}
    w = {m: tune_decision_weights(y, oof[m]) for m in GBDTS}

    pc_oof = [correct(oof[m], w[m]) for m in GBDTS]
    pc_test = [correct(test[m], w[m]) for m in GBDTS]

    # Recover the blend weights by refitting on OOF (same NM recipe as ensemble.py),
    # then apply the SAME weights to test.
    from scipy.optimize import minimize
    from sklearn.metrics import balanced_accuracy_score

    def blend(wv: NDArray, arrays: list[NDArray]) -> NDArray:
        s = np.exp(wv) / np.exp(wv).sum()
        return sum(s[i] * a for i, a in enumerate(arrays))

    res = minimize(
        lambda wv: (
            -float(balanced_accuracy_score(y, np.argmax(blend(wv, pc_oof), axis=1)))
        ),
        np.ones(len(GBDTS)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    blended_oof = blend(res.x, pc_oof)
    blended_test = blend(res.x, pc_test)
    dw = tune_decision_weights(y, blended_oof)
    oof_score = balanced_accuracy_score(y, weighted_predict(blended_oof, dw))
    print(
        f"reconstructed v1b OOF balanced_acc: {oof_score:.4f} (scored config was 0.9494)"
    )
    return weighted_predict(blended_test, dw)


def main() -> None:
    ds = load_dataset()
    v1b = champion_test_labels(ds.y)
    np.save(RESULTS_DIR / "champion_test_labels.npy", v1b)

    v2_csv = pl.read_csv(SUBMISSIONS_DIR / "ensemble_v2_bagmlp.csv").sort("id")
    l2i = {c: i for i, c in enumerate(CLASSES)}
    v2 = v2_csv["health_condition"].replace_strict(l2i).to_numpy()
    assert (v2_csv["id"].to_numpy() == ds.test_ids).all(), "row-order mismatch"

    diff = v1b != v2
    n = int(diff.sum())
    print(f"\ndisagreements: {n} / {len(v2)} = {100 * n / len(v2):.2f}% of test rows")
    print("(net LB effect of these flips: -0.0006)")

    print("\nclass transitions (v1b → v2):")
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            k = int(((v1b == a) & (v2 == b)).sum())
            if k:
                print(
                    f"  {CLASSES[a]:>9} → {CLASSES[b]:<9} {k:>7}  ({100 * k / n:.1f}% of flips)"
                )

    test_raw = pl.read_csv(DATA_DIR / "test.csv").sort("id")
    miss_key = (
        test_raw.select(pl.any_horizontal([pl.col(c).is_null() for c in KEY_DRIVERS]))
        .to_numpy()
        .ravel()
    )
    print(
        "\nkey-driver missingness among flipped rows: "
        f"{100 * miss_key[diff].mean():.1f}% (base rate {100 * miss_key.mean():.1f}%)"
    )

    adv_te_path = RESULTS_DIR / "adv_scores_test.npy"
    if adv_te_path.exists():
        adv_te = np.load(adv_te_path)
        print(
            "adv test-likeness of flipped rows: "
            f"median {np.median(adv_te[diff]):.3f} vs overall {np.median(adv_te):.3f}"
        )
    else:
        print("(no adv_scores_test.npy — skip adversarial characterization)")


if __name__ == "__main__":
    main()
