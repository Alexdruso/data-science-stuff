"""Split-half honest read of a blend, and the in-sample bias it corrects for.

    python src/blend_honest.py --arm s1 --stack
    python src/blend_honest.py --arm s1 --cross-surface s4

`ensemble.py` fits its blend weights with `optimize_blend_weights` **on the whole
OOF and scores on that same OOF**. The null it is priced against — `bag(lgbm x6)`
— is a parameter-free average. So the blend gets an optimistic bias that the null
does not, and the pre-registered breadth delta is `blend - bag`.

With 3-4 free parameters over 691k rows that bias should be far below 0.0001 and
irrelevant at the +0.0010 adopt bar. It is **not** irrelevant at the +0.0003
boundary between "flat" and "hold", which is decided at the fourth decimal. This
script measures the bias rather than assuming it:

  * **in-sample**  — fit on all rows, score on all rows (what `ensemble.py` prints)
  * **split-half** — fit on half the rows, score on the other half, both
    directions, averaged. No parameter is ever scored on the rows that chose it.

Both are reported against the same null so the delta can be read either way. This
is root CLAUDE.md lesson #3's shape ("fit the decision layer honestly, per arm"),
applied to a combiner rather than a threshold.

The halves are stratified on `y` and drawn with a fixed seed, so the number is
reproducible and the two arms of a paired read see identical splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import SPACES, bag_seeds, discover, resolve_arm, to_rank
from features import TARGET, build_features

from data_science_stuff.kaggle.blending import blend, optimize_blend_weights
from data_science_stuff.kaggle.io import competition_dirs
from data_science_stuff.kaggle.stacking import stack_oof

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)

#: The LightGBM family whose seed-bag is the null for every breadth read
#: (CLAUDE.md Day-6 pre-registration).
NULL_FAMILY = "s1"
SPLIT_SEED = 202608


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def split_half_blend(
    oofs: list[np.ndarray], y: np.ndarray, space: str
) -> tuple[float, np.ndarray]:
    """Mean held-out AUC over both halves, plus the full-data weights.

    Weights are refit on each half and scored on the complement, so the returned
    AUC contains no in-sample component. The weights handed back are the
    full-data ones — those are what a submission would actually deploy; only the
    *score* needs to be honest.
    """
    transform = SPACES[space]
    cols = [transform(o) for o in oofs]
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SPLIT_SEED)

    scores = []
    for fit_idx, score_idx in folds.split(np.zeros(len(y)), y):
        w = optimize_blend_weights(
            [c[fit_idx] for c in cols], y[fit_idx], lambda yy, p: auc(yy, p)
        )
        scores.append(auc(y[score_idx], blend([c[score_idx] for c in cols], w)))

    full_w = optimize_blend_weights(cols, y, lambda yy, p: auc(yy, p))
    return float(np.mean(scores)), full_w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="s1")
    parser.add_argument("--legs", default="")
    parser.add_argument("--cross-surface", default="")
    parser.add_argument("--stack", action="store_true")
    args = parser.parse_args()

    allowed = [args.arm, *[s for s in args.cross_surface.split(",") if s]]
    tags = args.legs.split(",") if args.legs else discover(allowed)
    for tag in tags:
        if resolve_arm(tag) not in allowed:
            msg = f"{tag} is on surface {resolve_arm(tag)!r}, not in {allowed}"
            raise ValueError(msg)

    # Row order comes from build_features, never from raw CSV order.
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy()
    oofs = [np.load(RESULTS_DIR / f"oof_{t}.npy").ravel() for t in tags]
    tests = [np.load(RESULTS_DIR / f"test_{t}.npy").ravel() for t in tags]
    for tag, oof in zip(tags, oofs):
        if len(oof) != len(y):
            msg = f"{tag}: OOF length {len(oof)} != {len(y)}"
            raise ValueError(msg)

    names, oofs, tests = bag_seeds(tags, oofs, tests)
    print(f"{len(names)} legs after bagging, surfaces {sorted({resolve_arm(t) for t in tags})}")
    for n, o in sorted(zip(names, oofs), key=lambda kv: -auc(y, kv[1])):
        print(f"  {auc(y, o):.6f}  {n}")

    null = next((o for n, o in zip(names, oofs) if n.startswith(NULL_FAMILY)), None)
    if null is None:
        msg = f"no {NULL_FAMILY} leg among {names} — nothing to price breadth against"
        raise ValueError(msg)
    null_auc = auc(y, null)
    print(f"\nNULL  bag(lgbm) = {null_auc:.6f}   bars: flat <+0.0003 | hold | adopt >=+0.0010")

    print(f"\n{'space':>6}  {'in-sample':>10}  {'split-half':>11}  "
          f"{'bias':>8}  {'honest delta':>13}  verdict")
    for space in SPACES:
        w = optimize_blend_weights(
            [SPACES[space](o) for o in oofs], y, lambda yy, p: auc(yy, p)
        )
        insample = auc(y, blend([SPACES[space](o) for o in oofs], w))
        honest, _ = split_half_blend(oofs, y, space)
        delta = honest - null_auc
        verdict = "ADOPT" if delta >= 0.0010 else ("hold" if delta >= 0.0003 else "flat")
        print(f"{space:>6}  {insample:>10.6f}  {honest:>11.6f}  "
              f"{insample - honest:>+8.6f}  {delta:>+13.6f}  {verdict}")

    if args.stack:
        # stack_oof runs its own outer CV, so its number is already honest and
        # needs no split-half correction — reported here for comparability.
        stack_o, _ = stack_oof(
            oofs,
            tests,
            y.astype(np.int64),
            lambda: LogisticRegression(C=1.0, max_iter=2000),
            use_logits=True,
        )
        s_auc = auc(y, stack_o[:, 1])
        d = s_auc - null_auc
        verdict = "ADOPT" if d >= 0.0010 else ("hold" if d >= 0.0003 else "flat")
        print(f"{'stack':>6}  {'—':>10}  {s_auc:>11.6f}  {'—':>8}  {d:>+13.6f}  {verdict}")

    print("\nrank corr vs the null leg:")
    null_r = to_rank(null)
    for n, o in zip(names, oofs):
        if not n.startswith(NULL_FAMILY):
            print(f"  {np.corrcoef(null_r, to_rank(o))[0, 1]:.4f}  {n}")


if __name__ == "__main__":
    main()
