"""Blend the legs on one surface and write a submission.

    python src/ensemble.py --arm s1
    python src/ensemble.py --arm s1 --legs s1,xgb_s1,cat_s1

AUC is rank-based, so a plain probability average is *not* scale-invariant: two
legs that rank identically but are calibrated differently will average badly.
Three blend spaces are therefore fitted and the best OOF one wins:

  * probability — the naive baseline
  * logit       — `stacking.clipped_logit`, the right space for averaging
                  well-calibrated probabilities
  * rank        — per-leg normalised ranks, fully calibration-invariant and
                  usually the strongest for an AUC metric

Two DIFFERENT invariants are easy to conflate here, and conflating them costs
the neural leg its only route to a submission:

  * **Paired arm-delta reads require the same surface.** That is root CLAUDE.md
    lesson #9: if you alter the train matrix, every OOF built that way reads
    systematically lower, so `s1 - s0` is only meaningful within one surface.
    `--arm` enforces this, and it is the default.
  * **Blending requires the same `y`, the same row order and comparable folds** —
    not the same feature matrix. An MLP on an imputed surface and a LightGBM on
    the NaN-native one both predict the same 691k labels in `build_features`
    order; blending them is precisely the model diversity a bag of GBDTs cannot
    manufacture for itself, and refusing it would make arms `s3`/`s4` pointless
    (their stated purpose is that an MLP cannot consume NaN).

So cross-surface blending is allowed, but only when asked for **by name** via
`--cross-surface`, never by discovery: a surface mix is the one error that
produces a perfectly plausible-looking wrong answer, because every arm's OOF has
length `len(y)` and no shape check can catch it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.blending import blend, optimize_blend_weights
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.stacking import clipped_logit

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)


def to_rank(p: np.ndarray) -> np.ndarray:
    return rankdata(p) / (len(p) + 1.0)


SPACES = {
    "prob": lambda p: p,
    "logit": clipped_logit,
    "rank": to_rank,
}


def resolve_arm(tag: str) -> str | None:
    """The surface a tag was produced on, or None if the tag isn't a leg.

    Tags are `<leg>_<arm>_s<seed>` (e.g. `xgb_s1_s42`) or `<arm>_s<seed>`
    (e.g. `s1_s42`) for the LightGBM arm itself.

    >>> resolve_arm("xgb_s1_s42"), resolve_arm("s0_s44"), resolve_arm("blend_x")
    ('s1', 's0', None)
    """
    base, _, seed = tag.rpartition("_s")
    if not base or not seed.isdigit():
        return None
    return base.rsplit("_", 1)[-1]


def discover(arms: list[str]) -> list[str]:
    """Every leg with both an OOF and a test array on one of these surfaces.

    `arms` is a list rather than a string only so an explicitly-requested
    cross-surface blend can be discovered; it is never widened implicitly. The
    match stays exact rather than a suffix test because a surface mix is the one
    error that produces a perfectly plausible-looking wrong answer — every arm's
    OOF has length len(y), so no shape check downstream can catch it.
    """
    tags = []
    for path in sorted(RESULTS_DIR.glob("oof_*.npy")):
        tag = path.stem[len("oof_") :]
        if resolve_arm(tag) not in arms:
            continue
        if (RESULTS_DIR / f"test_{tag}.npy").exists():
            tags.append(tag)
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--legs", default="")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--cross-surface",
        default="",
        help="extra surfaces whose legs may join the blend, e.g. 's3,s4'. "
             "Discovery never crosses surfaces on its own — see the module docstring.",
    )
    args = parser.parse_args()

    extra = [s for s in args.cross_surface.split(",") if s]
    allowed = [args.arm, *extra]

    tags = args.legs.split(",") if args.legs else discover(allowed)
    if not tags:
        print(f"no legs found for surface(s) {', '.join(allowed)}")
        return
    # Explicit even for hand-passed --legs: a surface mix is undetectable later,
    # because every arm's OOF has length len(y).
    for tag in tags:
        resolved = resolve_arm(tag)
        if resolved not in allowed:
            msg = (
                f"{tag} was produced on surface {resolved!r}, which is not among "
                f"{allowed} — pass it to --cross-surface if the mix is intended"
            )
            raise ValueError(msg)

    used = sorted({resolve_arm(t) for t in tags} - {None})
    if len(used) > 1:
        print(
            f"CROSS-SURFACE blend over {used} ({len(tags)} legs).\n"
            "  Valid because blending needs same y + same row order, not the same\n"
            "  feature matrix. NOT valid for a paired arm-delta read — do not quote\n"
            "  any number from this run as an arm effect."
        )
    else:
        print(f"all {len(tags)} legs resolve to surface {args.arm}")

    # Row order comes from build_features, never from raw CSV order.
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    y = train[TARGET].to_numpy()
    test_ids = test["id"].to_numpy()

    oofs = [np.load(RESULTS_DIR / f"oof_{t}.npy").ravel() for t in tags]
    tests = [np.load(RESULTS_DIR / f"test_{t}.npy").ravel() for t in tags]
    for tag, oof in zip(tags, oofs):
        if len(oof) != len(y):
            msg = f"{tag}: OOF length {len(oof)} != {len(y)} — surface mismatch?"
            raise ValueError(msg)

    print(f"arm {args.arm}: {len(tags)} legs")
    for tag, oof in sorted(zip(tags, oofs), key=lambda kv: -roc_auc_score(y, kv[1])):
        print(f"  {roc_auc_score(y, oof):.5f}  {tag}")

    print("\ncorrelation of leg OOF ranks (diversity):")
    ranks = np.column_stack([to_rank(o) for o in oofs])
    corr = np.corrcoef(ranks.T)
    print(f"  mean off-diagonal rank corr: {corr[np.triu_indices(len(tags), 1)].mean():.4f}"
          if len(tags) > 1 else "  (single leg)")

    best = (0.0, "", None, None)
    for space, fn in SPACES.items():
        o = [fn(a) for a in oofs]
        t = [fn(a) for a in tests]
        weights = optimize_blend_weights(o, y, lambda yy, p: float(roc_auc_score(yy, p)))
        auc = float(roc_auc_score(y, blend(o, weights)))
        print(f"\n{space:>6} blend OOF AUC {auc:.5f}  weights "
              f"{dict(zip(tags, np.round(weights, 3)))}")
        if auc > best[0]:
            best = (auc, space, weights, blend(t, weights))

    auc, space, weights, test_blend = best
    print(f"\nbest space: {space}  OOF AUC {auc:.5f}")
    name = args.name or f"blend_{args.arm}_{space}"

    # Submit ranks as probabilities is fine for AUC, but keep the file in [0,1].
    preds = to_rank(test_blend) if space == "rank" else test_blend
    preds = np.clip(preds, 0.0, 1.0)
    np.save(RESULTS_DIR / f"test_{name}.npy", preds)
    out = write_submission(SUBMISSIONS_DIR, f"{name}.csv", test_ids, TARGET, preds)
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
