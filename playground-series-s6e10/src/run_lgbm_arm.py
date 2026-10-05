"""Paired LGBM feature arms for PS S6E10 (PR-3, PR-4).

    python src/run_lgbm_arm.py --arm base        # must reproduce lgbm_pr1_base
    python src/run_lgbm_arm.py --arm orig        # PR-3: original as a feature source
    python src/run_lgbm_arm.py --arm te_pairs    # PR-4a: fold-safe TE on rating pairs
    python src/run_lgbm_arm.py --arm cat_ratings # PR-4b: ratings as categoricals

Every arm uses the PR-1 control's exact LGBM config (CPU, 8 threads, seed 42,
same folds), so its per-fold AUC is paired against `oof_lgbm_pr1_base.npy`.

Arms:
  orig        — no synthetic labels are used, so these are global features
                (leakage rule 2 territory, documented in CLAUDE.md):
                * `orig_pred`: an LGBM fit on the full ORIGINAL dataset, predicting
                  every synthetic train/test row;
                * `orig_rate_<col>`: the original's satisfaction rate for the row's
                  value of each categorical / rating column.
  te_pairs    — for each of the 78 rating pairs, a smoothed target rate of the
                (r_i, r_j) combination, fit INSIDE each fold on the training split
                (nested CV on train, `kaggle.encoding.add_fold_safe_target_encoding`).
  cat_ratings — the 13 0-5 ratings as `category` dtype (0 = N/A is not ordinal).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import RATING_COLS, TARGET
from matrices import ORIGINAL_FLAG, Matrices, build_matrices, load_original
from train_lgbm_orig import PARAMS

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.encoding import add_fold_safe_target_encoding
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)
CONTROL = "lgbm_pr1_base"
BAR = 0.0005

RATING_PAIRS = list(itertools.combinations(RATING_COLS, 2))
PAIR_COLS = [f"pair_{a}__{b}" for a, b in RATING_PAIRS]


def add_orig_features(m: Matrices) -> None:
    """PR-3: features computed from the original dataset only (no synthetic labels)."""
    assert m.X_orig is not None and m.y_orig is not None
    cols = [c for c in m.feature_cols if c != ORIGINAL_FLAG]
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=8,
    )
    model.fit(m.X_orig[cols], m.y_orig)
    orig = load_original().to_pandas()
    prior = float(m.y_orig.mean())
    for frame in (m.X, m.X_test):
        frame["orig_pred"] = model.predict_proba(frame[cols])[:, 1]
        for col in [*m.cat_cols, *RATING_COLS]:
            rates = orig.groupby(orig[col].astype(str))[TARGET].mean()
            frame[f"orig_rate_{col}"] = (
                frame[col].astype(str).map(rates).astype(float).fillna(prior)
            )


def add_pair_keys(m: Matrices) -> None:
    """PR-4a: integer key per rating pair (TE is fit later, inside each fold)."""
    for frame in (m.X, m.X_test):
        for (a, b), name in zip(RATING_PAIRS, PAIR_COLS):
            frame[name] = frame[a].astype(np.int16) * 10 + frame[b].astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "orig", "te_pairs", "cat_ratings"])
    args = parser.parse_args()

    m = build_matrices(with_original=args.arm == "orig")
    m.X, m.X_test = m.X.copy(), m.X_test.copy()
    if args.arm == "orig":
        add_orig_features(m)
    elif args.arm == "te_pairs":
        add_pair_keys(m)
    elif args.arm == "cat_ratings":
        for frame in (m.X, m.X_test):
            for col in RATING_COLS:
                frame[col] = pd.Categorical(frame[col], categories=list(range(6)))
    print(f"arm {args.arm}: train {m.X.shape}  test {m.X_test.shape}", flush=True)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        if args.arm == "te_pairs":
            x_tr, x_va, x_te = add_fold_safe_target_encoding(
                x_tr, y_tr, [x_va, x_te], PAIR_COLS, {1: "sat"}
            )
            # The raw integer keys stay out of the model: TE is the feature.
            x_tr, x_va, x_te = (f.drop(columns=PAIR_COLS) for f in (x_tr, x_va, x_te))
        model = LGBMClassifier(**PARAMS)
        model.fit(
            x_tr, y_tr, eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        print(
            f"  fold {fold}: best_iter={model.best_iteration_} "
            f"AUC {roc_auc_score(y_va, model.predict_proba(x_va)[:, 1]):.5f} "
            f"({time.time() - t0:.0f}s, {x_tr.shape[1]} feats)",
            flush=True,
        )
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(m.X, m.y, m.X_test, fit_fold, n_outputs=1, score_fn=fold_score)
    oof_auc = float(roc_auc_score(m.y, oof))
    name = f"lgbm_arm_{args.arm}"
    save_cv_result(RESULTS_DIR, name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{name}.npy", test)

    # Paired read against the PR-1 control, per fold on the same splits.
    from sklearn.model_selection import StratifiedKFold

    ctrl = np.load(RESULTS_DIR / f"oof_{CONTROL}.npy")
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(m.X, m.y)
    deltas = [
        roc_auc_score(m.y[va], oof[va]) - roc_auc_score(m.y[va], ctrl[va]) for _, va in splits
    ]
    d_oof = oof_auc - float(roc_auc_score(m.y, ctrl))
    n_pos = sum(d > 0 for d in deltas)
    verdict = "PASS" if d_oof >= BAR and n_pos >= 4 else "FAIL"
    print(f"\n{name}: OOF {oof_auc:.6f}  per-fold delta {np.round(deltas, 6).tolist()}")
    print(f"{name}: OOF delta {d_oof:+.6f}  positive folds {n_pos}/5  → {verdict}")
    print(f"corr with control {np.corrcoef(oof, ctrl)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
