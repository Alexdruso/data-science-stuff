"""Multinomial LR stacker on base-model OOFs — Deotte's meta-learner (PS S6E6).

Our ensemble.py uses Nelder-Mead on a single positive scalar weight per model →
a STAR-weak/GALAXY-strong NN gets ~0 weight because no scalar captures "use this
model's GALAXY column, ignore its STAR column." A multinomial logistic regression
on the base models' class-probability LOGITS has a weight per (model, class), so it
can extract complementary per-class strengths — exactly how cdeotte's 0.9702 stack
lifts a diverse-but-individually-weaker set. Honest 5-fold OOF stacking (fit on 4
folds of the OOF rows, predict the 5th); class_weight balanced; argmax-gated vs the
0.9659 Nelder-Mead blend.

Run:  python src/build_lr_stack.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from postprocess import optimize_thresholds, save_threshold_weights

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
# Diverse base set (skip near-duplicate lgbm variants that add only collinearity).
# Pruned to STRONG bases — ablation showed the 6 weak ~0.95 bases (mlp/mlp_la/realmlp/
# realmlp_emb/extratrees/rf) add only +0.0001 = dead weight / overfit surface.
MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
          "xgb_deotte", "realmlp_deotte", "catboost_deotte", "catboost_v3", "xgb_v3fe",
          "chain_cascade", "chain_cascade_xgb"]
SEEDS = [2024, 7, 13, 42, 99]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def main() -> None:
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())
    test_ids = te["id"].to_numpy()

    Ztr, Zte, used = [], [], []
    for m in MODELS:
        fo, ft = RESULTS_DIR / f"oof_{m}.npy", RESULTS_DIR / f"test_{m}.npy"
        if fo.exists() and ft.exists():
            Ztr.append(logit(np.load(fo)))
            Zte.append(logit(np.load(ft)))
            used.append(m)
    print(f"Stacking {len(used)} models: {used}")
    Ztr, Zte = np.hstack(Ztr), np.hstack(Zte)

    oof = np.zeros((len(y), 3))
    test = np.zeros((len(test_ids), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tri, vai in skf.split(Ztr, y):
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, n_jobs=-1)
            lr.fit(Ztr[tri], y[tri])
            oof[vai] += lr.predict_proba(Ztr[vai]) / len(SEEDS)
            test += lr.predict_proba(Zte) / (len(SEEDS) * 5)

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    tw, best = optimize_thresholds(oof, y)
    print(f"LR-stack OOF argmax: {argmax:.4f}  [Nelder-Mead blend 0.9655 / single-best 0.9654]")
    print(f"LR-stack OOF threshold-tuned: {best:.4f}  [blend thresh 0.9659]")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / "threshold_weights_lrstack.json")
    save_cv_result(RESULTS_DIR, "lrstack", [], best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / "oof_lrstack.npy", oof)
    np.save(RESULTS_DIR / "test_lrstack.npy", test)
    labels = le.inverse_transform(np.argmax(test * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, "lrstack.csv", test_ids, TARGET, labels)
    print("Saved → lrstack (oof/test/submission)")


if __name__ == "__main__":
    main()
