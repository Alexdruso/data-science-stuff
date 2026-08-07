"""Substrate check for the imputing surfaces `s3`/`s4` before the MLP eats them.

    python src/smoke_impute.py

`impute.py` has never executed end to end (CLAUDE.md item 9, "Correction to the
record": `results/` holds no `s3` array and `arm_scores.json` has no `s3` key).
`train_mlp.py` depends on one of these surfaces, and the MLP is the member the
Day-6 pre-registration named as most likely to carry the blend past +0.0010 —
so discovering at 18:00 that `s4` does not build would cost the day's headline
read. This runs the transforms on a subsample and asserts the invariants that
would otherwise fail silently or three hours late.

What is checked, and why each one is here rather than assumed:

1. **No NaN left in `IMPUTE_ORDER`.** An MLP cannot consume NaN at all. If the
   imputer declines a column, `train_mlp.to_matrix` median-fills it — which is
   fine, but it must be the documented fallback rather than an accident.
2. **Row order preserved.** The row-order invariant is the failure that once
   produced a 0.5-AUC submission in this repo. `_recompute_derived` rebuilds
   columns through polars and reassigns `.index`; this asserts the index really
   does come back identical.
3. **The budget constraint still holds.** `impute.py`'s own docstring records
   that the first smoke test produced `min(daily - sgw_lb) = -0.43`:
   freely-predicted parts can overshoot an *observed* `daily`.
   `_project_parts_onto_budget` exists to fix exactly that, and this is the
   assertion that says it worked.
4. **No inf, and finite column ranges.** `derive_v2` divides by columns that
   imputation can drive to 0, and an inf reaches the MLP as a NaN gradient.

This is a *substrate* check, not an arm-delta read: CLAUDE.md item 7 vetoed
scoring the imputation arms as LightGBM arms, and that veto stands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import impute
from run_arm import ARMS, assert_budget_constraint, build_matrices

SUBSAMPLE = 40_000
SEED = 42

#: Arms that actually fill `IMPUTE_ORDER`. `s2` (masks only) and `s5` (pooled
#: latent) deliberately leave NaN native — LightGBM consumes it — so asserting
#: no-NaN on them would fail a design decision rather than a bug. They are still
#: checked for row order, budget and inf.
IMPUTING_ARMS = {"s3", "s4"}


def check(arm: str, x: pd.DataFrame, y: np.ndarray, x_test: pd.DataFrame) -> bool:
    builder, transform, description = ARMS[arm]
    print(f"\n{'=' * 68}\n{arm}: {description}\n{'=' * 68}", flush=True)
    if transform is None:
        print("  no transform on this arm — nothing to smoke-test")
        return True

    tr_idx, va_idx = next(
        iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y))
    )
    x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
    before = {"tr": x_tr.index.to_numpy(), "va": x_va.index.to_numpy()}

    out_tr, out_va, out_te = transform(x_tr, y[tr_idx], x_va, x_test)
    print(f"  shapes  train {out_tr.shape}  val {out_va.shape}  test {out_te.shape}")

    ok = True

    # 1. columns the imputer claims to fill — only on arms that claim to
    if arm in IMPUTING_ARMS:
        for name, frame in (("train", out_tr), ("val", out_va), ("test", out_te)):
            left = {c: int(frame[c].isna().sum()) for c in impute.IMPUTE_ORDER}
            bad = {c: n for c, n in left.items() if n}
            if bad:
                print(f"  ✗ {name}: NaN remains in imputed columns {bad}")
                ok = False
        if ok:
            print(f"  ✓ no NaN left in {len(impute.IMPUTE_ORDER)} imputed columns")
    else:
        print("  – NaN-native arm by design; no-NaN check does not apply")

    # 2. row order — the invariant that once cost a 0.5-AUC submission
    for name, frame, orig in (
        ("train", out_tr, before["tr"]),
        ("val", out_va, before["va"]),
    ):
        if not np.array_equal(frame.index.to_numpy(), orig):
            print(f"  ✗ {name}: row order CHANGED by the transform")
            ok = False
    if ok:
        print("  ✓ row order preserved on every frame")

    # 3. the budget constraint — the bug impute.py's docstring records
    for name, frame in (("train", out_tr), ("val", out_va), ("test", out_te)):
        try:
            assert_budget_constraint(frame)
        except AssertionError as exc:
            print(f"  ✗ {name}: {exc}")
            ok = False
    if ok:
        slack = (
            out_tr.loc[
                out_tr["daily_screen_time_hours"].notna(), "daily_screen_time_hours"
            ]
            - out_tr.loc[out_tr["daily_screen_time_hours"].notna(), "sgw_lb"]
        ).min()
        print(f"  ✓ budget constraint holds (min daily − sgw_lb = {slack:+.6f})")

    # 4. inf / non-finite — reaches an MLP as a NaN gradient
    for name, frame in (("train", out_tr), ("val", out_va), ("test", out_te)):
        num = frame.select_dtypes(include=[np.number])
        n_inf = int(np.isinf(num.to_numpy(dtype=np.float64)).sum())
        if n_inf:
            cols = [c for c in num.columns if np.isinf(num[c].to_numpy()).any()]
            print(f"  ✗ {name}: {n_inf} inf values in {cols}")
            ok = False
    if ok:
        print("  ✓ no inf in any numeric column")

    # Context, not a pass/fail: how much the imputer actually changed.
    filled = {
        c: int(x_tr[c].isna().sum()) for c in impute.IMPUTE_ORDER if c in x_tr.columns
    }
    print(f"  (filled per column on the train split: {filled})")
    return ok


def main() -> None:
    builder = ARMS["s1"][0]
    x, y, x_test = build_matrices(builder)
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(x), size=min(SUBSAMPLE, len(x)), replace=False)
    keep.sort()
    x, y = x.iloc[keep], y[keep]
    x_test = x_test.iloc[: SUBSAMPLE // 2]
    print(f"subsample: train {x.shape}, test {x_test.shape}, base rate {y.mean():.4f}")

    results = {arm: check(arm, x, y, x_test) for arm in ("s2", "s3", "s4", "s5")}

    print(f"\n{'=' * 68}")
    for arm, ok in results.items():
        print(f"  {arm}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)
    print("all imputing surfaces build — the MLP is unblocked")


if __name__ == "__main__":
    main()
