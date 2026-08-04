"""Step 1 probe: is the label a smooth function of a recoverable latent index?

Evidence that motivated this (complete rows, 269,185 of them):
  * logistic regression on the 9 raw numerics reaches AUC 0.9277 vs LGBM 0.96312
  * its score ventiles run 0.09 -> 1.00 monotonically with no discontinuity, so
    the label is a noisy sigmoid of an index, NOT a hard threshold rule
  * standardised coefficients: social +1.28, daily +1.13, weekend +1.13,
    work -0.37, gaming -0.31, everything else ~0

Substituting the budget identity `daily = social + gaming + work + other` into
that index gives a concrete, testable candidate:

    social + daily + weekend - gaming - work  ==  2*social + other + weekend

i.e. "social media counts double, plus unclassified screen time, plus weekend
screen time". This probe tests that candidate and its neighbours directly, then
runs a small automated search, and measures everything against two reference
points: the 0.9277 linear ceiling and an LGBM fit on the same complete rows
(the achievable ceiling).

Pre-registered bar: a 1-D index with complete-row AUC >= 0.95 gets built as a
feature. Between 0.9277 and 0.95 -> note it, do not add it (a 127-leaf GBDT on
9 features almost certainly already represents it). Below 0.9277 -> discard.
HARD STOP: this block is timeboxed; it can redirect the day but must not
consume it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, _, _ = competition_dirs(__file__)

BAR_ADOPT = 0.95
LINEAR_CEILING = 0.9277

NUMERIC = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]


def main() -> None:
    frame = pl.read_csv(DATA_DIR / "train.csv").drop_nulls()
    y = frame[TARGET].to_numpy()
    g = {c: frame[c].to_numpy().astype(np.float64) for c in NUMERIC}
    print(f"complete rows: {frame.height}   positive rate {y.mean():.4f}\n")

    daily, social = g["daily_screen_time_hours"], g["social_media_hours"]
    gaming, work = g["gaming_hours"], g["work_study_hours"]
    weekend, sleep = g["weekend_screen_time"], g["sleep_hours"]
    other = daily - social - gaming - work
    eps = 1e-6

    # ── reference points ────────────────────────────────────────────────
    x_raw = np.column_stack([g[c] for c in NUMERIC])
    x_std = (x_raw - x_raw.mean(0)) / x_raw.std(0)
    lin = roc_auc_score(y, LogisticRegression(max_iter=2000).fit(x_std, y).decision_function(x_std))
    gbdt = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=127, verbose=-1, n_jobs=4
    ).fit(x_raw, y)
    gbdt_auc = roc_auc_score(y, gbdt.predict_proba(x_raw)[:, 1])
    print(f"reference  linear (logistic on 9 raw): {lin:.4f}")
    print(f"reference  LGBM in-sample (ceiling-ish): {gbdt_auc:.4f}")
    print(f"bar to adopt a 1-D index: {BAR_ADOPT}\n")

    # ── targeted candidates from the coefficient algebra ────────────────
    candidates: dict[str, np.ndarray] = {
        "2*social + other + weekend": 2 * social + other + weekend,
        "social + other + weekend": social + other + weekend,
        "social + other": social + other,
        "daily - gaming - work (= social+other)": daily - gaming - work,
        "social + daily + weekend - gaming - work": social + daily + weekend - gaming - work,
        "(social + other) / daily": (social + other) / (daily + eps),
        "(social + other) * weekend": (social + other) * weekend,
        "mean(daily, weekend) non-gaming": (daily + weekend) / 2 - gaming - work,
        "social + 0.5*other + weekend": social + 0.5 * other + weekend,
        "(2*social + other + weekend) / (24 - sleep)": (2 * social + other + weekend) / (24 - sleep + eps),
    }
    print("=== targeted candidates ===")
    for name, idx in sorted(candidates.items(), key=lambda kv: -roc_auc_score(y, kv[1])):
        print(f"  {roc_auc_score(y, idx):.4f}  {name}")

    # ── small automated search: w1*social + w2*other + w3*weekend ───────
    print("\n=== weight search over  a*social + b*other + c*weekend ===")
    best = (0.0, None)
    grid = np.arange(0.0, 3.01, 0.25)
    for a, b, c in itertools.product(grid, grid, grid):
        if a + b + c == 0:
            continue
        auc = roc_auc_score(y, a * social + b * other + c * weekend)
        if auc > best[0]:
            best = (auc, (a, b, c))
    print(f"  best {best[0]:.4f} at (social, other, weekend) = {best[1]}")

    # ── does adding the winner to the raw set close the linear->GBDT gap? ──
    winner = max(
        [*candidates.items(), ("weight-search winner", best[1][0] * social + best[1][1] * other + best[1][2] * weekend)],
        key=lambda kv: roc_auc_score(y, kv[1]),
    )
    w_auc = roc_auc_score(y, winner[1])
    print(f"\nbest 1-D index: {w_auc:.4f}  ({winner[0]})")
    print(f"  vs linear ceiling {lin:.4f}  -> {'BEATS' if w_auc > lin else 'below'} it")
    print(f"  vs adopt bar {BAR_ADOPT}     -> {'ADOPT' if w_auc >= BAR_ADOPT else 'DO NOT ADOPT'}")

    aug = np.column_stack([x_raw, winner[1]])
    aug_gbdt = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=127, verbose=-1, n_jobs=4
    ).fit(aug, y)
    aug_auc = roc_auc_score(y, aug_gbdt.predict_proba(aug)[:, 1])
    print(
        f"\nin-sample LGBM with the index added: {aug_auc:.4f} "
        f"(vs {gbdt_auc:.4f} without, Δ {aug_auc - gbdt_auc:+.5f})"
    )
    print("  NOTE: in-sample only — this ranks the idea, it does not measure it.")
    print("  A real read requires the CV arm; that is what run_arm.py is for.")


if __name__ == "__main__":
    main()
