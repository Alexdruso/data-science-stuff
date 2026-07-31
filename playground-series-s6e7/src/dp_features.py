"""Day-23: label-free DRIVER-POSTERIOR feature block (`_dp`).

The block is `P(driver | x_obs)` for all three missing drivers, plus the three
rule-band masses of the sleep posterior:

    sleep_duration          11 cols  (SLEEP_EDGES bins, rule cuts 6/7 forced as edges)
    stress_level             3 cols  (low / medium / high)
    physical_activity_level  3 cols  (sedentary / moderate / active)
    dp_sleep_band            3 cols  (P(sleep<6), P(6<=sleep<7), P(sleep>=7))
                            --------
                            20 cols

WHY THIS AND NOT `imp_features` (Day-17). That block reconstructs the WEAKEST of the
three drivers (`sleep_duration`, R^2 0.151) as a scalar point estimate. Day-22's
`train_sm.py` diagnostic found `physical_activity_level` is by far the most recoverable
(logloss gain +0.5515, top-1 0.712 vs 0.339 prior) and it has never been fed to a strong
leg as features. The `dp_sleep_band` columns are the rule's own coordinate in probability
form — a tree gets that from splits for free, a smooth net cannot synthesise it from an
11-way posterior, which is the whole Rung-2-parallel argument that made te13 pay.

PRIOR ART it must be distinct from, stated before the run: Day-7 `train_dp.py` fed 3
transductive driver posteriors (+9 cols) to an LGBM and measured FAIL (solo -0.0002,
blend +0.0001). That verdict is PRE-te13, REPAIRED-surface, TREE-consumer. This is the
same "pre-TE-era veto re-earned on the deployed recipe" pattern that moved FTT
0.9478 -> 0.9504 and TabM 0.9489 -> 0.9504 (both then failed their seat tests honestly).

LEAK-SAFETY. The imputers predict a DRIVER, never the label, and were fitted on
train+test pooled observed rows — the construction `train_dp.py` was accepted under.
No label touches them, so there is nothing to cross-fit and the block is fold-INVARIANT
(hence no (seed, fold) cache key — unlike the te/knn/imp blocks).

⚠️ DELIBERATELY NOT reusing `imp_features.imp_block_for_fold`: its cache key
`imp_{tag}_s{seed}_f{fold}.npz` does not encode the driver, so a second driver would
collide byte-for-byte with the 20 existing `imp_u_*` entries and silently serve stale
sleep blocks — the identical bug class `_kseg()` was added for on Day-15.

⚠️ HONEST CAVEAT on the motivating number. The +0.5515 / 0.712 diagnostic is IN-SAMPLE
and fitted on train+test POOLED. Activity's mask is also trigger-coupled in train and
uniform in test (P(NaN | smoking='occasional') = 0.161 train vs 0.054 test, the Day-4
mechanism shift), which sits directly on this channel. The realistic effect is smaller
than the diagnostic suggests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, Dataset

# Must match train_sm.py's construction exactly — these arrays are its cached output.
DRIVERS = ["sleep_duration", "stress_level", "physical_activity_level"]
DRIVER_K = {"sleep_duration": 11, "stress_level": 3, "physical_activity_level": 3}
STRESS = ["low", "medium", "high"]
ACT = ["sedentary", "moderate", "active"]
# train_sm.SLEEP_EDGES = [-inf, 4.5, 5.25, 5.7, 6.0, 6.35, 6.7, 7.0, 7.6, 8.3, 9.2, inf]
# digitize(x, EDGES[1:-1]) -> bins 0..10, so bins 0..3 are sleep < 6.0, bins 4..6 are
# [6.0, 7.0) and bins 7..10 are >= 7.0 — the rule's two cuts are exact bin boundaries.
SLEEP_BANDS = [(0, 4), (4, 7), (7, 11)]
BAND_NAMES = ["dp_sleep_lt6", "dp_sleep_6to7", "dp_sleep_ge7"]


def dp_block(ds: Dataset) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str]]:
    """Return (train_block, test_block, names) — fold-invariant, label-free."""
    n_tr, n_te = len(ds.train), len(ds.test)
    cols: list[NDArray[np.float64]] = []
    names: list[str] = []
    sleep_post: NDArray[np.float64] | None = None

    for drv in DRIVERS:
        cache = RESULTS_DIR / f"sm_imp_{drv}.npy"
        if not cache.exists():
            raise SystemExit(
                f"missing {cache} — build it with src/train_sm.py's _fit_imputers first"
            )
        p = np.load(cache)
        k = DRIVER_K[drv]
        # The row order is pd.concat([ds.train, ds.test]) in build_features() sorted
        # order (train_sm.py:_fit_imputers). Assert it rather than trust it: a silent
        # off-by-n_tr here would misalign every row (row-order invariant, CLAUDE.md).
        assert p.shape == (n_tr + n_te, k), f"{cache.name} shape {p.shape} != {(n_tr + n_te, k)}"
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), f"{cache.name} rows do not sum to 1"
        if drv == "sleep_duration":
            sleep_post = p
        cols.append(p)
        names += [f"dp_{drv}_{i}" for i in range(k)]

    assert sleep_post is not None
    bands = np.stack([sleep_post[:, lo:hi].sum(axis=1) for lo, hi in SLEEP_BANDS], axis=1)
    assert np.allclose(bands.sum(axis=1), 1.0, atol=1e-6), "sleep bands do not partition"
    cols.append(bands)
    names += BAND_NAMES

    full = np.concatenate(cols, axis=1).astype(np.float32)
    assert full.shape[1] == len(names) == 20, f"block width {full.shape[1]} != 20"
    assert np.isfinite(full).all(), "non-finite value in the dp block"
    return full[:n_tr], full[n_tr:], names
