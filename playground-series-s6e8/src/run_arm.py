"""Run one null-handling arm at several seeds, and compare arms pairwise.

    python src/run_arm.py --arm s1 --seeds 42,43,44
    python src/run_arm.py --compare

Every arm shares the fold assignment at a given seed (same `y`, same
`StratifiedKFold(seed)`), so arm-vs-arm deltas are *paired* — the fold-split
noise cancels instead of adding.

DEVICE IS LOCKED to CPU for the whole sweep and must not be changed mid-sweep.
Measured on this box: one fold takes 53.8s on `cuda` vs 22.2s on `cpu`, and the
two paths give AUC 0.96247 vs 0.96217 with best_iter 903 vs 608. That 0.0003
device gap is the same size as the effects this sweep is trying to resolve, so
mixing devices across arms would swamp the measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import impute
import te_cross
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features
from features_signal import build_features_signal
from features_v2 import RAW_CATEGORICAL_COLS, build_features_v2

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

N_FOLDS = 5
SCORES_PATH = RESULTS_DIR / "arm_scores.json"

# Locked — see the module docstring.
LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": 12,
    "device_type": "cpu",
}

Builder = Callable[[pl.DataFrame], pl.DataFrame]
Transform = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]

ARMS: dict[str, tuple[Builder, Transform | None, str]] = {
    "s0": (build_features, None, "v1 reference, NaN native"),
    "s1": (build_features_v2, None, "GATE: budget-constraint features, no imputation"),
    "s2": (build_features_v2, impute.add_masks, "s1 + mask indicators (control)"),
    "s3": (build_features_v2, impute.median_impute, "s1 + median/mode impute + masks"),
    "s4": (
        build_features_v2,
        impute.relational_impute,
        "s1 + constraint-aware relational impute + masks",
    ),
    "s5": (build_features_v2, impute.latent_impute, "s1 + pooled latent intensity + masks"),
    "s6": (
        build_features_signal,
        None,
        "s1 minus the 7 noise columns and their derivatives (15 features)",
    ),
    "s7": (
        build_features_v2,
        te_cross.add_te_crosses,
        "s1 + fold-safe TE of 6 two-way quantile-bin crosses",
    ),
    "s8": (
        build_features_v2,
        te_cross.add_te_singles,
        "s1 + fold-safe TE of 8 single quantile-binned columns (q=64)",
    ),
}


def build_matrices(builder: Builder) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Feature matrices for one arm. Row order is owned by `build_features`."""
    train = builder(pl.read_csv(DATA_DIR / "train.csv"))
    test = builder(pl.read_csv(DATA_DIR / "test.csv"))

    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    train_pd, test_pd = train.to_pandas(), test.to_pandas()
    for col in RAW_CATEGORICAL_COLS:
        if col not in train_pd.columns:
            continue  # arm s6 drops the categoricals entirely
        levels = pd.api.types.union_categoricals(
            [train_pd[col].astype("category"), test_pd[col].astype("category")]
        ).categories
        train_pd[col] = pd.Categorical(train_pd[col], categories=levels)
        test_pd[col] = pd.Categorical(test_pd[col], categories=levels)

    y = train_pd[TARGET].to_numpy()
    return train_pd[feature_cols], y, test_pd[feature_cols]


def assert_budget_constraint(x: pd.DataFrame) -> None:
    """`sgw_lb <= daily` is a verified invariant — a breach means a bug."""
    if "sgw_lb" not in x.columns:
        return
    both = x["daily_screen_time_hours"].notna()
    slack = (x.loc[both, "daily_screen_time_hours"] - x.loc[both, "sgw_lb"]).min()
    if slack < -1e-6:
        msg = f"budget constraint violated: min(daily - sgw_lb) = {slack}"
        raise AssertionError(msg)


def run_seed(arm: str, seed: int) -> float:
    builder, transform, _ = ARMS[arm]
    x, y, x_test = build_matrices(builder)
    assert_budget_constraint(x)

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if transform is not None:
            # Fit-on-train-split-only happens in here, never outside.
            x_tr, x_va, x_te = transform(x_tr, y_tr, x_va, x_te)
        params = {**LGBM_PARAMS, "random_state": seed}
        model = LGBMClassifier(**params)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    oof, test_pred, fold_scores = run_cv(
        x,
        y,
        x_test,
        fit_fold,
        n_splits=N_FOLDS,
        seed=seed,
        n_outputs=1,
        score_fn=lambda yv, p: float(roc_auc_score(yv, p)),
    )
    auc = float(roc_auc_score(y, oof))
    tag = f"{arm}_s{seed}"
    np.save(RESULTS_DIR / f"oof_{tag}.npy", oof)
    np.save(RESULTS_DIR / f"test_{tag}.npy", test_pred)
    save_cv_result(RESULTS_DIR, tag, fold_scores, auc, metric_name="auc")
    # NOTE: pre-transform width. Arms s2-s5 add mask/impute columns inside
    # fit_fold (fold-aware), so they train on more columns than this shows.
    print(f"  {tag}: base_features={x.shape[1]:>3}  OOF AUC {auc:.5f}", flush=True)
    return auc


def record(arm: str, seed: int, auc: float) -> None:
    scores: dict[str, dict[str, float]] = {}
    if SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text())
    scores.setdefault(arm, {})[str(seed)] = auc
    SCORES_PATH.write_text(json.dumps(scores, indent=2, sort_keys=True))


def compare(baseline: str = "s0") -> None:
    """Paired deltas vs the baseline arm, on the seeds both arms actually ran."""
    if not SCORES_PATH.exists():
        print("no scores yet")
        return
    scores: dict[str, dict[str, float]] = json.loads(SCORES_PATH.read_text())
    base = scores.get(baseline, {})
    print(f"paired deltas vs {baseline} (3-seed paired read resolves ~0.0003)\n")
    print(f"{'arm':<5} {'seeds':<6} {'mean OOF':>10} {'Δ vs ' + baseline:>12} "
          f"{'sd(Δ)':>9}  description")
    for arm, per_seed in sorted(scores.items()):
        shared = sorted(set(per_seed) & set(base), key=int)
        mean = float(np.mean([per_seed[s] for s in sorted(per_seed, key=int)]))
        if arm == baseline or not shared:
            print(f"{arm:<5} {len(per_seed):<6} {mean:>10.5f} {'—':>12} {'—':>9}  "
                  f"{ARMS[arm][2] if arm in ARMS else ''}")
            continue
        deltas = np.array([per_seed[s] - base[s] for s in shared])
        sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")
        print(f"{arm:<5} {len(shared):<6} {mean:>10.5f} {deltas.mean():>+12.5f} "
              f"{sd:>9.5f}  {ARMS[arm][2] if arm in ARMS else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare()
        return
    if not args.arm:
        parser.error("--arm is required unless --compare")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"arm {args.arm}: {ARMS[args.arm][2]}", flush=True)
    for seed in seeds:
        record(args.arm, seed, run_seed(args.arm, seed))
    compare()


if __name__ == "__main__":
    main()
