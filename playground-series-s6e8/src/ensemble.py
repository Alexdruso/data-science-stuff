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

**Seeds of one model are bagged before weights are fitted** (`--bag`, on by
default). Six LightGBM seeds are not six legs — they are one leg measured six
times, and feeding them to Nelder-Mead as separate columns both overweights the
family that happens to have the most seeds and spends the optimizer's budget on
collinear directions. Averaging them first also makes the *right* null explicit:
the question "does a second model family pay?" must be asked against
`bag(lgbm x 6)`, never against a single seed, because seed-bagging already buys
the variance reduction that a second family would otherwise be credited for
(root CLAUDE.md lesson #1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features

from data_science_stuff.kaggle.blending import blend, optimize_blend_weights
from data_science_stuff.kaggle.io import competition_dirs, write_submission
from data_science_stuff.kaggle.stacking import clipped_logit, stack_oof

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


def family(tag: str) -> str:
    """The model+surface a tag belongs to, with the seed stripped.

    >>> family("xgb_s1_s42"), family("s1_s47"), family("mlp_s3_s42")
    ('xgb_s1', 's1', 'mlp_s3')
    """
    base, _, seed = tag.rpartition("_s")
    return base if base and seed.isdigit() else tag


def bag_seeds(
    tags: list[str], oofs: list[np.ndarray], tests: list[np.ndarray]
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Average every seed of one model into a single leg.

    Probabilities, not ranks: seeds of one model share a calibration, so their
    mean is itself a probability. See the module docstring for why this has to
    happen before weights are fitted.
    """
    groups: dict[str, list[int]] = {}
    for i, tag in enumerate(tags):
        groups.setdefault(family(tag), []).append(i)

    names, bag_oofs, bag_tests = [], [], []
    for name, idx in groups.items():
        names.append(f"{name}x{len(idx)}" if len(idx) > 1 else name)
        bag_oofs.append(np.mean([oofs[i] for i in idx], axis=0))
        bag_tests.append(np.mean([tests[i] for i in idx], axis=0))
    return names, bag_oofs, bag_tests


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
    parser.add_argument(
        "--no-bag",
        action="store_true",
        help="treat every seed as its own leg (default: average seeds within a model first)",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="also fit an LR-on-logits meta-learner (stacking.stack_oof) and report it",
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

    print(f"arm {args.arm}: {len(tags)} raw arrays")
    for tag, oof in sorted(zip(tags, oofs), key=lambda kv: -roc_auc_score(y, kv[1])):
        print(f"  {roc_auc_score(y, oof):.5f}  {tag}")

    if not args.no_bag:
        tags, oofs, tests = bag_seeds(tags, oofs, tests)
        print(f"\nbagged to {len(tags)} legs (seeds averaged within a model):")
        for tag, oof in sorted(zip(tags, oofs), key=lambda kv: -roc_auc_score(y, kv[1])):
            print(f"  {roc_auc_score(y, oof):.5f}  {tag}")

    # A single family after bagging is not a blend — it is the bag itself, and
    # it is the null every added family has to beat. Write it and stop.
    if len(tags) == 1:
        name = args.name or f"bag_{tags[0]}"
        auc = float(roc_auc_score(y, oofs[0]))
        print(f"\nonly one leg after bagging — writing the bag itself, OOF AUC {auc:.5f}")
        preds = np.clip(tests[0], 0.0, 1.0)
        np.save(RESULTS_DIR / f"test_{name}.npy", preds)
        out = write_submission(SUBMISSIONS_DIR, f"{name}.csv", test_ids, TARGET, preds)
        print(f"submission -> {out}")
        return

    print("\ncorrelation of leg OOF ranks (diversity):")
    ranks = np.column_stack([to_rank(o) for o in oofs])
    corr = np.corrcoef(ranks.T)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            print(f"  {corr[i, j]:.4f}  {tags[i]} ~ {tags[j]}")
    print(f"  mean off-diagonal rank corr: {corr[np.triu_indices(len(tags), 1)].mean():.4f}")

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

    if args.stack:
        # An LR meta on logits: the alternative combiner to try when the scalar
        # blend plateaus. `stack_oof` runs its own multi-seed outer CV, so the
        # number it returns is honest rather than in-sample.
        stack_o, stack_t = stack_oof(
            oofs,
            tests,
            y.astype(np.int64),
            lambda: LogisticRegression(C=1.0, max_iter=2000),
            use_logits=True,
        )
        stack_auc = float(roc_auc_score(y, stack_o[:, 1]))
        print(f"\n stack (LR on logits) OOF AUC {stack_auc:.5f}")
        if stack_auc > best[0]:
            best = (stack_auc, "stack", None, stack_t[:, 1])

    auc, space, weights, test_blend = best
    best_leg = max(float(roc_auc_score(y, o)) for o in oofs)
    print(f"\nbest space: {space}  OOF AUC {auc:.5f}")
    print(f"best single leg  OOF AUC {best_leg:.5f}   combiner delta {auc - best_leg:+.5f}")
    name = args.name or f"blend_{args.arm}_{space}"

    # Submit ranks as probabilities is fine for AUC, but keep the file in [0,1].
    preds = to_rank(test_blend) if space == "rank" else test_blend
    preds = np.clip(preds, 0.0, 1.0)
    np.save(RESULTS_DIR / f"test_{name}.npy", preds)
    out = write_submission(SUBMISSIONS_DIR, f"{name}.csv", test_ids, TARGET, preds)
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
