"""Exact-generation-rule features (user idea, 2026-07-10 night; discussion/717222).

broccoli beef proved the original dataset's label is an EXACT depth-4 tree on
(sleep_duration [thresholds 6 AND 7], stress_level, physical_activity_level):

  sleep<6:  stress==high -> unhealthy, else -> at-risk
  sleep>=6: stress!=low -> at-risk
            stress==low & activity!=active -> at-risk
            stress==low & activity==active: sleep>=7 -> fit, else -> at-risk

The lever no prior FE had: THREE-VALUED DEDUCTION on partially-missing rows —
e.g. stress=medium forces at-risk through every branch, sleep and activity
unneeded. For each row we emit the SET of labels reachable over all completions
of its missing drivers ('ar', 'fit|ar', 'ar|unh', ...) — a noise-free logical
prior that resolves many "missing-driver" rows exactly, targeting the
at-risk<->minority boundary where all residual error lives.

Note for the record: train_fe's combo TE used sleep_QUALITY; the true rule
triple is sleep_DURATION-buckets x stress x activity.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

STRESS = ["low", "medium", "high"]
ACT = ["sedentary", "moderate", "active"]
MISSING = -1


def _leaf(b: int, s: str, a: str) -> str:
    """Exact rule label for sleep bucket b (0: <6, 1: [6,7), 2: >=7), stress s, activity a."""
    if b == 0:
        return "unhealthy" if s == "high" else "at-risk"
    if s != "low":
        return "at-risk"
    if a != "active":
        return "at-risk"
    return "fit" if b == 2 else "at-risk"


def _label_set(b: int, s: int, a: int) -> str:
    """Reachable-label set over all completions of missing (-1) drivers."""
    bs = [b] if b != MISSING else [0, 1, 2]
    ss = [STRESS[s]] if s != MISSING else STRESS
    as_ = [ACT[a]] if a != MISSING else ACT
    labels = sorted({_leaf(bb, sss, aaa) for bb, sss, aaa in product(bs, ss, as_)})
    return "|".join(labels)


# 4x4x4 partial patterns -> label-set string, precomputed once
_TABLE: dict[tuple[int, int, int], str] = {
    (b, s, a): _label_set(b, s, a)
    for b in (MISSING, 0, 1, 2)
    for s in (MISSING, 0, 1, 2)
    for a in (MISSING, 0, 1, 2)
}


def rule_label_set(df: pd.DataFrame) -> pd.Series:
    """Per-row reachable-label-set string ('at-risk', 'at-risk|fit', ...)."""
    sleep = df["sleep_duration"]
    b = np.where(
        sleep.isna(), MISSING, np.where(sleep < 6, 0, np.where(sleep < 7, 1, 2))
    ).astype(int)
    s = df["stress_level"].map({v: i for i, v in enumerate(STRESS)}).fillna(MISSING)
    a = (
        df["physical_activity_level"]
        .map({v: i for i, v in enumerate(ACT)})
        .fillna(MISSING)
    )
    keys = list(zip(b, s.astype(int), a.astype(int)))
    return pd.Series([_TABLE[k] for k in keys], index=df.index, name="rule_set")


def add_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """df + rule_set (categorical) + rule_determined (0/1)."""
    out = df.copy()
    rs = rule_label_set(df)
    out["rule_set"] = rs.astype("category")
    out["rule_determined"] = (~rs.str.contains(r"\|")).astype("float32")
    return out
