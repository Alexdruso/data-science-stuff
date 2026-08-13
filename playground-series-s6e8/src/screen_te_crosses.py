"""Screen quantile-bin TE crosses by held-out residual correction (Day 10).

Pre-registered in CLAUDE.md "Next steps" item 10. The question is NOT "does a TE
arm score better" — that costs hours per candidate. It is the strictly prior
question: does `bag(xgb x6)` leave *uncaptured* structure in a candidate's cells?

Instrument: split train rows by `id % 2`. On half A estimate the count-shrunk
mean residual `E[y - oof | cell]`; add it to `oof` on half B, measure the AUC
change on B. Symmetrise. Cell means see A's labels only, so B is honest.

Bar: >= +0.0010 mean held-out delta -> build the real TE arm. Below -> do not.

Calibration (root CLAUDE.md lesson #2), both on our own data:
  * positive control: the exact-integer-`daily` cells (measured OOF residual
    -0.07..-0.20). The instrument MUST register this, or it cannot detect
    real residual structure.
  * negative control: random cells with the cell-size distribution matched to
    each candidate. Must read ~0; that is the noise band.

Row-order invariant: `y` comes through `build_features_v2` (which sorts by
SORT_KEY), matching how every `oof_*.npy` was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import TARGET  # noqa: E402
from features_v2 import build_features_v2  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"

#: Pre-registered candidate crosses. Each is (name, col_a, col_b, n_bins).
CANDIDATES: list[tuple[str, str, str, int]] = [
    ("social_share x daily", "social_share", "daily_screen_time_hours", 16),
    ("n_obs_sgw x sgw_lb", "n_obs_sgw", "sgw_lb", 16),
    ("screen_block_nulls x sgw_lb", "screen_block_nulls", "sgw_lb", 16),
    ("awake_share x sleep", "screen_share_of_awake", "sleep_hours", 16),
    ("gaming_share x social_share", "gaming_share", "social_share", 16),
    ("stress_ord x daily", "stress_ord", "daily_screen_time_hours", 16),
]

#: Shrinkage denominator for the per-cell residual mean (same spirit as the
#: `smooth=20` default in `kaggle.encoding.add_fold_safe_target_encoding`).
SMOOTH = 20.0


def qcodes(x: NDArray[np.float64], fit: NDArray[np.bool_], q: int) -> NDArray[np.int64]:
    """Quantile-bin codes with edges from `fit` rows only; NaN -> its own code."""
    vals = x[fit & ~np.isnan(x)]
    if vals.size < 2:
        return np.zeros(len(x), dtype=np.int64)
    uniq = np.unique(vals)
    # A low-cardinality column (n_obs_sgw, stress_ord, ...) is its own binning.
    if uniq.size <= q:
        codes = np.searchsorted(uniq, x)
        codes[np.isnan(x)] = uniq.size
        return codes.astype(np.int64)
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, q + 1)[1:-1]))
    codes = np.searchsorted(edges, x, side="right")
    codes[np.isnan(x)] = len(edges) + 1
    return codes.astype(np.int64)


def cells_for(
    df: pl.DataFrame, a: str, b: str, q: int, fit: NDArray[np.bool_]
) -> NDArray[np.int64]:
    """Cross-product cell id for two quantile-binned columns."""
    xa = df[a].to_numpy().astype(float)
    xb = df[b].to_numpy().astype(float)
    ca, cb = qcodes(xa, fit, q), qcodes(xb, fit, q)
    return ca * (cb.max() + 1) + cb


def screen(
    cell: NDArray[np.int64],
    y: NDArray[np.int64],
    oof: NDArray[np.float64],
    half: NDArray[np.bool_],
) -> tuple[float, int]:
    """Mean symmetrised held-out AUC delta from the cell residual correction."""
    resid = y - oof
    n_cells = int(cell.max()) + 1
    deltas = []
    for fit_mask in (half, ~half):
        score_mask = ~fit_mask
        cnt = np.bincount(cell[fit_mask], minlength=n_cells).astype(float)
        tot = np.bincount(cell[fit_mask], weights=resid[fit_mask], minlength=n_cells)
        # count-shrunk toward 0 (the global residual mean of an OOF is ~0)
        adj = tot / (cnt + SMOOTH)
        corrected = np.clip(oof[score_mask] + adj[cell[score_mask]], 0.0, 1.0)
        deltas.append(
            roc_auc_score(y[score_mask], corrected) - roc_auc_score(y[score_mask], oof[score_mask])
        )
    occupied = int((np.bincount(cell, minlength=n_cells) > 0).sum())
    return float(np.mean(deltas)), occupied


def matched_random_cells(
    cell: NDArray[np.int64], rng: np.random.Generator
) -> NDArray[np.int64]:
    """Random cell assignment with the same cell-size distribution."""
    sizes = np.bincount(cell)
    sizes = sizes[sizes > 0]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    rng.shuffle(labels)
    return labels.astype(np.int64)


def corruption_control(
    cell: NDArray[np.int64],
    y: NDArray[np.int64],
    oof: NDArray[np.float64],
    half: NDArray[np.bool_],
    rng: np.random.Generator,
    sd: float,
) -> tuple[float, float]:
    """Positive control: inject known per-cell offsets, see if the screen recovers them.

    The integer-`daily` cells are the WRONG positive control — that structure was
    already measured worthless in AUC terms (+0.000392 with perfect labels), so a
    null reading there says nothing about the instrument. Instead, damage `oof`
    with a per-cell offset of known magnitude. That is genuine, cell-shaped,
    recoverable structure by construction, so the screen MUST register it.

    Returns:
        ``(damage, recovered)`` — the AUC lost to the corruption, and the delta
        the screen reports when run on the corrupted predictions.
    """
    offsets = rng.normal(0.0, sd, size=int(cell.max()) + 1)
    bad = np.clip(oof + offsets[cell], 0.0, 1.0)
    damage = roc_auc_score(y, bad) - roc_auc_score(y, oof)
    recovered, _ = screen(cell, y, bad, half)
    return float(damage), recovered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leg", default="xgb_s1", help="OOF leg to bag as the baseline")
    p.add_argument("--seed", type=int, default=219, help="negative-control RNG seed")
    args = p.parse_args()

    train = build_features_v2(pl.read_csv(HERE / "data/train.csv"))
    y = train[TARGET].to_numpy().astype(np.int64)
    ids = train["id"].to_numpy()

    files = sorted(RESULTS.glob(f"oof_{args.leg}_s*.npy"))
    if not files:
        raise SystemExit(f"no oof_{args.leg}_s*.npy in {RESULTS}")
    oof = np.mean([np.load(f) for f in files], axis=0)
    if len(oof) != len(y):
        raise SystemExit(f"length mismatch: oof {len(oof)} vs y {len(y)}")

    half = (ids % 2).astype(bool)
    base = roc_auc_score(y, oof)
    print(f"baseline: bag({args.leg} x{len(files)}) OOF AUC = {base:.6f}")
    print(f"halves: A={(~half).sum()} B={half.sum()}   shrink={SMOOTH:g}")
    print("\nBAR (pre-registered): >= +0.0010 mean held-out delta -> build the arm\n")
    print(f"{'candidate':<30} {'cells':>7} {'delta':>10} {'neg ctrl':>10} {'verdict':>9}")
    print("-" * 70)

    rng = np.random.default_rng(args.seed)
    fit_all = np.ones(len(y), dtype=bool)

    # --- positive control: recover a known per-cell corruption --------------
    ref_cell = cells_for(train, "social_share", "daily_screen_time_hours", 16, fit_all)
    pos_delta = 0.0
    for sd in (0.02, 0.05, 0.10):
        damage, recovered = corruption_control(ref_cell, y, oof, half, rng, sd)
        frac = recovered / -damage if damage < 0 else float("nan")
        print(f"{'[+ctrl] cell noise sd=' + f'{sd:g}':<30} {'':>7} {recovered:>+10.6f} "
              f"{'':>10}  damage {damage:+.6f} recovers {frac:>5.1%}")
        pos_delta = max(pos_delta, recovered)
    print()

    rows = []
    for name, a, b, q in CANDIDATES:
        cell = cells_for(train, a, b, q, fit_all)
        delta, occ = screen(cell, y, oof, half)
        neg = screen(matched_random_cells(cell, rng), y, oof, half)[0]
        verdict = "BUILD" if delta >= 0.0010 else "no"
        rows.append((name, occ, delta, neg, verdict))
        print(f"{name:<30} {occ:>7} {delta:>+10.6f} {neg:>+10.6f} {verdict:>9}")

    print("-" * 70)
    best = max(rows, key=lambda r: r[2])
    print(f"best candidate: {best[0]}  delta {best[2]:+.6f}")
    print(f"positive control best recovery {pos_delta:+.6f} "
          f"({'DETECTED — instrument valid' if pos_delta >= 0.0010 else 'FAILED — instrument invalid'})")
    negs = np.array([r[3] for r in rows])
    print(f"negative-control band: {negs.min():+.6f} .. {negs.max():+.6f}")
    print(f"\nVERDICT: {'BUILD the TE arm' if best[2] >= 0.0010 else 'DO NOT BUILD — bar not cleared'}")


if __name__ == "__main__":
    main()
