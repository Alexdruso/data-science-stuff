"""M-slope screen for a surface change that landed in the "hold" band.

    python src/screen_surface.py --challenger s1 --stock s0 --seeds 42,43,44,45,46,47

The pre-registered adoption bar is +0.0010. A candidate between +0.0003 and
+0.0010 is *detectable* — the seed-to-seed sd of OOF AUC here is only ~0.0001 —
but detectable is not deployable. Root CLAUDE.md lesson #1: a variance reducer
is significantly positive at every single seed by construction, so per-seed
significance cannot tell information from noise removal. Only the decay
exponent can.

Read the exponent `b` from `|delta| ~ M**b`:
  * `b > 0`      information — averaging *reveals* it; survives at deployed breadth
  * `b ~ -0.5`   selection noise (the winner's curse)
  * `b ~ -1`     variance reduction — dead at any real breadth

Two things this script does that are easy to skip and wrong to skip:

1. **A no-fit AUC decision layer.** `honest_weighted_layer` (the default) fits
   class weights for an argmax decision — that is a balanced-accuracy artifact.
   AUC is rank-based and has no decision layer at all, so the layer here just
   scores.
2. **Calibrating the null band on THIS data** via `selection_null_arms`, per
   lesson #2. The selection-noise band is instrument-dependent and must never
   be imported from another competition.

Needs >= 6 members per arm: lesson #4 — s6e7 published a +0.475 exponent from a
3-point fit and had to correct it to -0.158 at six points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.screening import (
    classify_exponent,
    screen_mechanism,
    selection_null_arms,
)

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

MIN_MEMBERS = 6


def auc_layer(
    _y_fit: np.ndarray,
    _proba_fit: np.ndarray,
    y_eval: np.ndarray,
    proba_eval: np.ndarray,
) -> float:
    """No-fit scorer: AUC is rank-based, so there is nothing to fit."""
    return float(roc_auc_score(y_eval, proba_eval[:, 1]))


def load_members(arm: str, seeds: list[int]) -> np.ndarray:
    """(n_members, n_rows, 2) probabilities for one arm."""
    members = []
    for seed in seeds:
        path = RESULTS_DIR / f"oof_{arm}_s{seed}.npy"
        if not path.exists():
            msg = f"missing {path.name} — run `run_arm.py --arm {arm} --seeds {seed}`"
            raise FileNotFoundError(msg)
        p = np.load(path).ravel().astype(np.float64)
        members.append(np.column_stack([1.0 - p, p]))
    return np.stack(members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--stock", default="s0")
    parser.add_argument("--seeds", default="42,43,44,45,46,47")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < MIN_MEMBERS:
        print(
            f"WARNING: {len(seeds)} members. Lesson #4 says do not fit an exponent on "
            f"fewer than ~{MIN_MEMBERS} points — s6e7's +0.475 became -0.158 at six."
        )

    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    challenger = load_members(args.challenger, seeds)
    stock = load_members(args.stock, seeds)
    print(f"challenger {args.challenger} {challenger.shape}   stock {args.stock} {stock.shape}")

    per_seed = [
        roc_auc_score(y, challenger[i, :, 1]) - roc_auc_score(y, stock[i, :, 1])
        for i in range(len(seeds))
    ]
    print(f"per-seed paired deltas: {np.round(per_seed, 5).tolist()}")
    print(f"mean {np.mean(per_seed):+.5f}   sd {np.std(per_seed, ddof=1):.5f}\n")

    # Lesson #2: calibrate the selection-noise band on THIS data before quoting it.
    #
    # CAUTION: `selection_null_arms` splits the members into two DISJOINT halves,
    # so a null arm has len(seeds)//2 members and the null exponent is fit on
    # that many points — not on len(seeds). Passing 6 seeds yields a 3-point
    # null fit, which lesson #4 forbids, and the resulting band is ~3 exponent
    # units wide and swallows every possible result. A usable 6-point null needs
    # TWELVE stock members.
    print("calibrating the selection-noise null band on this data ...", flush=True)
    null_ch, null_st = selection_null_arms(stock, y.astype(np.int64), layer=auc_layer)
    if len(null_ch) < MIN_MEMBERS:
        print(
            f"  UNDER-POWERED: null arms have {len(null_ch)} members "
            f"(= len(seeds)//2), so the exponent is a {len(null_ch)}-point fit. "
            f"Supply {2 * MIN_MEMBERS} stock seeds for a usable band; until then the "
            "band below is not evidence and the verdict must rest on the two robust "
            "anchors (b~-0.5, b~-1) and on the observed retention."
        )
    null_screen = screen_mechanism(
        null_ch, null_st, y.astype(np.int64), layer=auc_layer, max_size=len(null_ch), n_jobs=4
    )
    lo = null_screen.exponent - 2 * null_screen.exponent_sd
    hi = null_screen.exponent + 2 * null_screen.exponent_sd
    print(f"null arm exponent {null_screen.exponent:+.3f} ± {null_screen.exponent_sd:.3f}")
    print(f"selection-noise band: ({lo:+.3f}, {hi:+.3f})\n")

    screen = screen_mechanism(
        challenger,
        stock,
        y.astype(np.int64),
        layer=auc_layer,
        max_size=len(seeds),
        null_band=(lo, hi),
        n_jobs=4,
    )
    print(screen)
    print(f"\nexponent {screen.exponent:+.3f} ± {screen.exponent_sd:.3f}")
    print(f"printed verdict: {classify_exponent(screen.exponent, null_band=(lo, hi))}")

    # The band-free read. Retention is measured, not fitted, so it stands even
    # when the null calibration is under-powered.
    d1, dm = float(screen.deltas[0]), float(screen.deltas[-1])
    m = int(screen.sizes[-1])
    print(f"\n--- band-free read (valid even with an unusable null band) ---")
    print(f"  retention at M={m}: {dm:+.6f} / {d1:+.6f} = {dm / d1:.0%}")
    print(f"  if variance reduction (~1/M):   would be {d1 / m:+.6f}  "
          f"(observed is {dm / (d1 / m):.1f}x larger)")
    print(f"  if selection noise (~M^-0.5):   would be {d1 / m**0.5:+.6f}  "
          f"(observed is {dm / (d1 / m**0.5):.1f}x larger)")
    print(f"  se on the observed M={m} delta: ±{float(screen.stderr[-1]):.6f}")


if __name__ == "__main__":
    main()
