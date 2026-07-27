"""Day-21 follow-up: how does the OOF SELECTION BIAS dilute with breadth?

Day-20 (`probe_oof_selection`) measured realmlp's selection-on-the-scored-rows optimism
at **+0.000186 at ONE member** and closed with an explicit, unmeasured caveat:

    "ONE fold, ONE seed per leg. The deployed legs are 8-12-seed BREADTH averages. Each
     seed selects its own epoch on its own fold split, so the tilt lands on a DIFFERENT
     random subset of rows per seed; averaging probabilities dilutes each tilt by 1/S
     while BA's argmax threshold is unchanged. The breadth leg's bias is therefore <= the
     single-seed +0.000186, probably materially less. This is the largest residual
     uncertainty and it points toward a SMALLER number."

That caveat is load-bearing: the whole cross-leg LEVEL comparison behind the #2 finals
menu ("realmlp 0.9507 solo vs honest hgbc 0.9505 / lgbm 0.9506") sits inside it, and
Day-20 had to record "realmlp is the strongest solo leg is NOT established at this
resolution".

`probe_snap_breadth` (today) established that per-member effects decay ~1/M under member
averaging. The selection tilt is exactly such a per-member effect, so the same curve
should apply — and it can be measured for FREE from that probe's cached per-epoch val
probabilities, with no new GPU.

PROTOCOL (reuses `results/cache/snap_breadth/epochs_f1_m*.npy`, one fold, 8 members that
share a split and TE block so their val rows are averageable):
  DEPLOYED(M)  each member picks its best epoch on A u B, average the M members'
               probs, score BA on A u B          <- what the oof_*.npy arrays are
  HONEST(M)    each member picks its epoch on half A, average, score on half B
               (and B -> A), averaged over 20 stratified halvings   <- unbiased
  BIAS(M) = DEPLOYED(M) - HONEST(M)

No adoption decision hangs on this, so there are no bars. What it can change:
  * BIAS(8) << BIAS(1)  =>  the deployed breadth arrays are nearly honest, the Day-20
    caveat resolves in the direction it predicted, and the cross-leg level reads (hence
    "realmlp is the strongest solo leg") are re-established.
  * BIAS(8) ~ BIAS(1)   =>  the tilt does NOT dilute, the caveat hardens, and every
    realmlp-vs-GBDT level comparison in the log needs a correction term.

Run: python src/probe_bias_dilution.py  ->  results/probe_bias_dilution.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from probe_snap_breadth import AB_SEED, FOLD, N_SPLITS, SPLIT_SEED_BASE, fold_indices
from train_common import RESULTS_DIR, load_dataset

MEMBERS = [42, 7, 123, 2024, 99, 2025, 31, 7777]  # the deployed 8-seed breadth


def bacc(y: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, np.argmax(probs, axis=1)))


def main() -> None:
    ds = load_dataset()
    y = ds.y
    _tr, va_idx = fold_indices(y, SPLIT_SEED_BASE, FOLD)
    y_va = y[va_idx]

    cache = RESULTS_DIR / "cache" / "snap_breadth"
    members = []
    for s in MEMBERS:
        p = cache / f"epochs_f{FOLD}_m{s}.npy"
        if not p.exists():
            raise SystemExit(f"missing {p} — run probe_snap_breadth.py first")
        members.append(np.load(p))

    splits = [
        next(iter(StratifiedKFold(2, shuffle=True, random_state=AB_SEED + r).split(
            np.zeros(len(y_va)), y_va)))
        for r in range(N_SPLITS)
    ]

    # deployed: every member picks its best epoch on the FULL val fold (what ships)
    full_pick = [int(np.argmax([bacc(y_va, P[e]) for e in range(len(P))])) for P in members]
    # honest: every member picks on the selection half only
    half_pick: dict[tuple[int, int, int], int] = {}
    for mi, P in enumerate(members):
        for rep, (a, b) in enumerate(splits):
            for side, sel in enumerate((a, b)):
                half_pick[(mi, rep, side)] = int(
                    np.argmax([bacc(y_va[sel], P[e][sel]) for e in range(len(P))])
                )

    lines = [
        "=" * 78,
        "probe_bias_dilution — does the OOF selection bias dilute with breadth?",
        f"fold {FOLD}, u surface, members {MEMBERS} (shared split + TE block)",
        f"DEPLOYED(M) = pick on A u B, score A u B  |  HONEST(M) = pick on A, score B "
        f"(both ways, {N_SPLITS} halvings)",
        "Day-20 measured BIAS(1) = +0.000186 and flagged the dilution as its largest "
        "open caveat.",
        "=" * 78,
        "",
        "  M   DEPLOYED    HONEST      BIAS = DEP-HON   (se)      share of BIAS(1)",
    ]
    print("\n".join(lines))

    bias1 = None
    for m in range(1, len(members) + 1):
        dep = bacc(
            y_va, np.mean([members[mi][full_pick[mi]] for mi in range(m)], axis=0)
        )
        hon: list[float] = []
        for rep, (a, b) in enumerate(splits):
            for side, (_sel, sco) in enumerate(((a, b), (b, a))):
                probs = np.mean(
                    [members[mi][half_pick[(mi, rep, side)]][sco] for mi in range(m)],
                    axis=0,
                )
                hon.append(bacc(y_va[sco], probs))
        h = np.asarray(hon)
        bias = dep - h.mean()
        if bias1 is None:
            bias1 = bias
        se = h.std(ddof=1) / np.sqrt(len(h))
        row = (
            f"  {m}   {dep:.6f}   {h.mean():.6f}   {bias:+.6f}       "
            f"({se:.6f})   {bias / bias1:5.2f}"
        )
        lines.append(row)
        print(row)

    lines += [
        "",
        f"HEADLINE  BIAS(1) = {bias1:+.6f}  (Day-20 independent read: +0.000186)",
    ]
    dep8 = lines[-2]
    _ = dep8
    out = "\n".join(lines)
    (RESULTS_DIR / "probe_bias_dilution.txt").write_text(out + "\n")
    print(f"\nwritten -> {RESULTS_DIR / 'probe_bias_dilution.txt'}")


if __name__ == "__main__":
    main()
