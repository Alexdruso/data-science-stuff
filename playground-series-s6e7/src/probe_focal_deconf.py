"""Split-half deconfound of the focal1.0 ladder read (Day-17).

focal1.0 read dw +0.00060 with argmax -0.00873 — the distorted-surface
signature under which the ladder's IN-SAMPLE dw instrument (weights tuned and
scored on the same val rows) inflates (Day-14 honest-meta lesson: in-sample-dw
confound removed -> gbdt_h -5e-5, nn_h -2.2e-4). This reruns base and focal1.0
on the exact fold-1 design and scores decision weights OUT-OF-SAMPLE:
fit w on half the val rows, score the other half, both directions x 3 seeds
(12 half-scores per arm). If focal's paired delta survives, it is real;
if it shrinks/flips, the +0.0006 was instrument inflation.

Run: python src/probe_focal_deconf.py -> results/probe_focal_deconf.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, RESULTS_DIR
from tune_realmlp_td import BASE_CFG, FOLD_SEED, build_fold1_design

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

ARMS: dict[str, dict[str, object]] = {
    "base": {},
    "focal1.0": {"focal_gamma": 1.0},
}


def val_probs(overrides, X_tr, X_val, y_tr, y_val, cat_names):  # noqa: ANN001, ANN201
    from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

    cfg = {**BASE_CFG, **overrides, "random_state": FOLD_SEED}
    torch.manual_seed(FOLD_SEED)
    np.random.seed(FOLD_SEED)  # noqa: NPY002 — library-internal RNG expects it
    model = RealMLP_TD_Classifier(**cfg)
    model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_names)
    proba = model.best_val_probs_.astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return proba


def main() -> None:
    X_tr, X_val, y_tr, y_val, cat_names = build_fold1_design()
    probs = {}
    for arm, overrides in ARMS.items():
        cache = RESULTS_DIR / "cache" / f"focal_deconf_{arm.replace('.', '')}.npy"
        if cache.exists():
            probs[arm] = np.load(cache)
            print(f"[{arm}] cached", flush=True)
            continue
        probs[arm] = val_probs(overrides, X_tr, X_val, y_tr, y_val, cat_names)
        np.save(cache, probs[arm])
        am = balanced_accuracy_score(y_val, probs[arm].argmax(axis=1))
        print(f"[{arm}] argmax {am:.5f}", flush=True)
    assert probs["base"].shape == (len(y_val), N_CLASSES)

    halves = []
    for seed in range(3):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        a, b = next(sss.split(y_val, y_val))
        halves += [(a, b), (b, a)]

    scores = {arm: [] for arm in ARMS}
    for fit_idx, eval_idx in halves:
        for arm, p in probs.items():
            w = np.asarray(tune_decision_weights(y_val[fit_idx], p[fit_idx]))
            scores[arm].append(
                balanced_accuracy_score(
                    y_val[eval_idx], weighted_predict(p[eval_idx], w)
                )
            )
    lines = ["focal1.0 split-half dw deconfound (fold-1 design, 6 half-cells)"]
    base = np.array(scores["base"])
    for arm, s in scores.items():
        s = np.array(s)
        d = s - base
        lines.append(
            f"  {arm:9s} oos-dw {s.mean():.5f}  vs base {d.mean():+.5f} "
            f"(positive {int((d > 0).sum())}/{len(d)})"
        )
    ins_base = balanced_accuracy_score(
        y_val,
        weighted_predict(
            probs["base"], np.asarray(tune_decision_weights(y_val, probs["base"]))
        ),
    )
    ins_focal = balanced_accuracy_score(
        y_val,
        weighted_predict(
            probs["focal1.0"],
            np.asarray(tune_decision_weights(y_val, probs["focal1.0"])),
        ),
    )
    lines.append(
        f"  in-sample reference: base {ins_base:.5f}  focal {ins_focal:.5f} "
        f"({ins_focal - ins_base:+.5f} — the ladder's read)"
    )
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_focal_deconf.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
