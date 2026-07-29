"""Day-22 Phase 3: STRUCTURED MARGINALIZATION of the missing driver (`sm` leg).

See CLAUDE.md "PRE-REGISTRATION #5" Phase 3. This is the one lever class veto #18 and
screening rule #3 both explicitly leave open: a leg that differs **in kind**, not a
per-member improvement of an existing one.

The label is an exact depth-4 rule on (sleep_duration, stress_level,
physical_activity_level) plus cell-clustered generator noise. 25.85% of rows have >=1 of
those three drivers masked, and they carry 86% of all residual error. A discriminative
model must learn `P(y | x_obs)` on those rows *from those rows*. But the problem
FACTORISES, and each factor is learnable where it is densely observed:

    P(y | x_obs) = sum_{b,s,a}  P(bin=b | x_obs) P(stress=s | x_obs) P(act=a | x_obs)
                                * P(y | b, s, a)

  * `P(y | b, s, a)` -- the cell posterior -- is fitted on the 511,675 rows where all
    three drivers ARE observed. Labels are used, so this is fitted fold-safely.
  * `P(driver | x_obs)` -- the three imputers -- are fitted where that driver IS
    observed, which is 88-95% of rows, and they use **no labels at all**, so they may
    legitimately be fitted on train and test together (this is a real advantage of the
    factorised form: the imputers see 986k rows, and the composition never has to be
    learned).

Sleep is binned rather than banded: the rule cuts at 6 and 7 are forced bin edges, but
the generator modelled sleep continuously (701 uniques, a sigmoid at the cuts), so extra
quantile edges inside each band let the cell table keep the within-band gradient.

DISTINCTNESS FROM PRIOR ART (pre-registered before building -- if it is not distinct,
this is the same channel and is already dead):
  * `probe_miss_tta` (FLAT) marginalised over the **mask** at predict time using the
    model's own P(y|x_obs). It introduced no new estimator; this introduces four.
  * Day-2's marginalisation used the **marginal** P(d), not P(d | x_obs).
  * Day-17 `probe_impute_driver` (+0.00015 solo, seat vetoes #14/#17) fed a **point
    estimate** E[sleep|x_obs] as a *feature* -- the model still had to learn the mixture
    -- and predicted the **value** (R^2 = 0.151) rather than the discrete cell the rule
    actually consumes.

Independence of the missing drivers given x_obs is assumed; it only bites on the 7.0% of
missing-driver rows with >=2 drivers masked.

Run:  python src/train_sm.py     ->  results/{oof,test}_sm[_<tag>].npy
Env:  S6E7_SM_SEEDS (default 42,7,123)   S6E7_RUN_TAG   S6E7_SM_ALPHA (cell smoothing)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, Dataset, bagged_cv, finalize, load_dataset

DRIVERS = ["sleep_duration", "stress_level", "physical_activity_level"]
STRESS = ["low", "medium", "high"]
ACT = ["sedentary", "moderate", "active"]
SEEDS = [int(s) for s in os.environ.get("S6E7_SM_SEEDS", "42,7,123").split(",")]
ALPHA = float(os.environ.get("S6E7_SM_ALPHA", "20"))
N_CLASSES = 3

# Rule cuts at 6 and 7 are forced edges; the rest are quantiles inside each band so the
# cell table keeps the generator's within-band sigmoid.
SLEEP_EDGES = np.array([-np.inf, 4.5, 5.25, 5.7, 6.0, 6.35, 6.7, 7.0, 7.6, 8.3, 9.2, np.inf])
N_BINS = len(SLEEP_EDGES) - 1


def _imputer_matrix(df: pd.DataFrame, drop: str) -> pd.DataFrame:
    """Feature frame for imputing `drop`: every other raw column, cats as category dtype."""
    X = df.drop(columns=[drop]).copy()
    for c in X.columns:
        if X[c].dtype == object or isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def _fit_imputers(ds: Dataset) -> tuple[dict[str, NDArray[np.float64]], int]:
    """Label-free driver posteriors P(driver | x_obs) for every train row, then test rows.

    Fitted on train+test pooled (no labels are involved, so this leaks nothing), on the
    rows where the driver is observed; predicted for all rows. Cached to disk.
    """
    n_tr = len(ds.train)
    full = pd.concat([ds.train, ds.test], ignore_index=True)
    sleep_bin_all = np.digitize(full["sleep_duration"].to_numpy(dtype=float), SLEEP_EDGES[1:-1])
    targets = {
        "sleep_duration": (sleep_bin_all, N_BINS),
        "stress_level": (full["stress_level"].map({v: i for i, v in enumerate(STRESS)}).to_numpy(), 3),
        "physical_activity_level": (
            full["physical_activity_level"].map({v: i for i, v in enumerate(ACT)}).to_numpy(),
            3,
        ),
    }
    out: dict[str, NDArray[np.float64]] = {}
    report: list[str] = []
    for drv, (tgt, k) in targets.items():
        cache = RESULTS_DIR / f"sm_imp_{drv}.npy"
        obs = ~full[drv].isna().to_numpy()
        if cache.exists():
            out[drv] = np.load(cache)
        else:
            t0 = time.time()
            X = _imputer_matrix(full, drv)
            clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=100,
                l2_regularization=1.0,
                categorical_features="from_dtype",
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
            clf.fit(X[obs], tgt[obs].astype(int))
            proba = np.zeros((len(full), k), dtype=np.float64)
            proba[:, clf.classes_] = clf.predict_proba(X)
            np.save(cache, proba)
            out[drv] = proba
            print(f"  imputer {drv}: k={k}  {time.time() - t0:.0f}s")
        p = out[drv]
        prior = np.bincount(tgt[obs].astype(int), minlength=k) / obs.sum()
        ll = log_loss(tgt[obs].astype(int), p[obs], labels=list(range(k)))
        ll0 = log_loss(
            tgt[obs].astype(int), np.tile(prior, (int(obs.sum()), 1)), labels=list(range(k))
        )
        acc = float((p[obs].argmax(1) == tgt[obs]).mean())
        report.append(
            f"  {drv:26s} k={k:2d}  in-sample logloss {ll:.4f} vs prior {ll0:.4f} "
            f"(gain {ll0 - ll:+.4f})  top1 {acc:.4f} vs {prior.max():.4f}"
        )
    print("driver-imputer quality (in-sample on observed rows):")
    for r in report:
        print(r)
    return out, n_tr


def _driver_posteriors(
    ds: Dataset,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    full_p, n_tr = _fit_imputers(ds)
    return {d: p[:n_tr] for d, p in full_p.items()}, {d: p[n_tr:] for d, p in full_p.items()}


def _hard_or_soft(
    df: pd.DataFrame, soft: NDArray[np.float64], col: str, levels: list[str] | None
) -> NDArray[np.float64]:
    """Point mass where the driver is observed, the imputer posterior where it is not."""
    q = soft.copy()
    if levels is None:  # sleep bins
        obs = ~df["sleep_duration"].isna().to_numpy()
        idx = np.digitize(df["sleep_duration"].to_numpy(dtype=float), SLEEP_EDGES[1:-1])
    else:
        obs = ~df[col].isna().to_numpy()
        idx = df[col].map({v: i for i, v in enumerate(levels)}).to_numpy()
    q[obs] = 0.0
    q[obs, idx[obs].astype(int)] = 1.0
    return q


def _cell_table(
    y: NDArray[np.int64], qb: NDArray[np.float64], qs: NDArray[np.float64], qa: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Smoothed P(y | bin, stress, act) from rows where all three drivers are point masses."""
    hard = (qb.max(1) == 1.0) & (qs.max(1) == 1.0) & (qa.max(1) == 1.0)
    b, s, a = qb[hard].argmax(1), qs[hard].argmax(1), qa[hard].argmax(1)
    flat = (b * 3 + s) * 3 + a
    counts = np.zeros((N_BINS * 9, N_CLASSES), dtype=np.float64)
    np.add.at(counts, (flat, y[hard]), 1.0)
    prior = counts.sum(0) / counts.sum()
    tab = (counts + ALPHA * prior) / (counts.sum(1, keepdims=True) + ALPHA)
    return tab.reshape(N_BINS, 3, 3, N_CLASSES)


def _compose(
    tab: NDArray[np.float64],
    qb: NDArray[np.float64],
    qs: NDArray[np.float64],
    qa: NDArray[np.float64],
    chunk: int = 100_000,
) -> NDArray[np.float64]:
    out = np.empty((len(qb), N_CLASSES), dtype=np.float64)
    for i in range(0, len(qb), chunk):
        j = min(i + chunk, len(qb))
        m = np.einsum("is,ia,bsac->ibc", qs[i:j], qa[i:j], tab, optimize=True)
        out[i:j] = np.einsum("ib,ibc->ic", qb[i:j], m, optimize=True)
    return out / out.sum(1, keepdims=True)


def main() -> None:
    ds = load_dataset()
    tr_p, te_p = _driver_posteriors(ds)

    qb = _hard_or_soft(ds.train, tr_p["sleep_duration"], "sleep_duration", None)
    qs = _hard_or_soft(ds.train, tr_p["stress_level"], "stress_level", STRESS)
    qa = _hard_or_soft(ds.train, tr_p["physical_activity_level"], "physical_activity_level", ACT)
    tb = _hard_or_soft(ds.test, te_p["sleep_duration"], "sleep_duration", None)
    ts = _hard_or_soft(ds.test, te_p["stress_level"], "stress_level", STRESS)
    ta = _hard_or_soft(ds.test, te_p["physical_activity_level"], "physical_activity_level", ACT)

    def fit_fold(tr_idx, val_idx, seed, fold):  # type: ignore[no-untyped-def]
        tab = _cell_table(ds.y[tr_idx], qb[tr_idx], qs[tr_idx], qa[tr_idx])
        return _compose(tab, qb[val_idx], qs[val_idx], qa[val_idx]), _compose(tab, tb, ts, ta)

    oof, test, scores = bagged_cv(ds, fit_fold, seeds=SEEDS)
    finalize("sm", ds, oof, test, scores)


if __name__ == "__main__":
    main()
