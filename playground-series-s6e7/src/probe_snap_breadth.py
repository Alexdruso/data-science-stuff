"""Day-21 pre-registration #3: does `snapshot_k=4` survive BREADTH AVERAGING?

Context.  Final #2 (`realmlp_te13_u_breadth`) is a SOLO leg — no combiner — so veto
#18's absorption mechanism does not apply and a leg-level recipe delta reaches the
deliverable by construction.  Day-20's `probe_oof_selection` Experiment C measured
epoch-selection rules HONESTLY (select on half A, score on half B, 40 splits) and
found snapshot averaging positive and monotone in k:

    snap2 +0.000174 / snap3 +0.000234 / snap4 +0.000325   (39/40 splits positive)

That is the largest honestly-measured un-run delta left on the board.  It has never
been run as a real 5-fold arm.

The question this probe answers is NOT "is snap4 better than bacc-selection at one
member" — that is already measured.  It is: **does the advantage survive averaging
over 4 members?**  This matters because snapshot averaging IS a variance-reduction
mechanism and so is breadth averaging; if they substitute, the deployed 8-seed
breadth has already collected the gain and there is nothing to swap in.

⚠️  WHY NOT JUST COMPARE dw-OOF LEVELS OF TWO FULL 5-FOLD ARMS.  The deployed OOF of
a bacc-selected arm is inflated by max-over-epochs selection on the very rows it is
scored on (+0.000186, `probe_oof_selection`); a snap-k arm averages the top-k epochs
and is inflated LESS.  So a naive paired dw-OOF read is biased DOWNWARD against snap4
by roughly the inflation gap — that is exactly how Day-10 recorded snapshot averaging
as a "wash".  This probe therefore uses the SAME honest instrument that produced the
+0.000325 headline, extended to averaged members.

PROTOCOL (fold 1, u surface, deployed te+te13 recipe from cache — the
`probe_oof_selection.run_realmlp` protocol, extended to M members)
  * ONE fold split and ONE TE block (seed 42, fold 1) shared by every member, so the
    members' val rows are identical and can be averaged per row.  Members differ ONLY
    in the model's `random_state` (42, 7, 123, 2024 → the deployed breadth's first
    four seeds).  Per-epoch val probabilities are captured for each member.
  * 20 stratified A/B halvings of the val fold.  For each split and each rule:
      select the epoch(s) on half A, form that member's val probs, average the M
      members' probs, score balanced accuracy on half B (and B→A), average.
    => HONEST for both arms; the selection bias cancels by construction.
  * Reported at M = 1, 2, 3, 4 so the shrinkage TRAJECTORY is visible, plus per-member
    M=1 deltas so one lucky member cannot carry the mean.

PRE-REGISTERED BARS on the honest (snap4 − bacc) delta at M=4:
  >= +0.00015  and sign-consistent  => the advantage survives averaging; ESCALATE to
                a real 5-fold `S6E7_SNAP=4` arm (seeds 42/7/123/2024, ~1.5 h GPU),
                then to N=8 and a #2 swap only under the Lane-A escalation rules
                (pairs MC > deployed #2 by more than noise, submission slot,
                finals_audit pass, byte verification).
  <= +0.00005  => snapshot averaging and breadth averaging substitute; STOP.
  otherwise    => INCONCLUSIVE, no swap.  No post-hoc bar moving.

HONEST PRIOR, stated before the run: snapshot averaging reduces the variance of a
single member's val posterior, and averaging 4 members reduces the same variance
component.  If that is the whole mechanism, the M=4 delta should be MATERIALLY below
+0.000325 and this probe stops the lane.  The campaign's own record agrees: 6/6
deltas escalated to a more-averaged instrument shrank ~50%.  Expect ~+0.00016, i.e.
INCONCLUSIVE between the bars.

Run: python src/probe_snap_breadth.py   ->  results/probe_snap_breadth.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset
from zoo_common import te13_block_for_fold

import os

# Default = the pre-registered matched-N=4 canary. `S6E7_SNAP_MEMBERS` extends the SAME
# instrument to the deployed breadth's full 8 seeds — the M value that actually governs
# the deliverable. Per-member epoch captures are cached, so extending never refits.
MEMBER_SEEDS = [
    int(s)
    for s in os.environ.get("S6E7_SNAP_MEMBERS", "42,7,123,2024").split(",")
    if s.strip()
]
FOLD = 1
SPLIT_SEED_BASE = 42  # fold split comes from the seed-42 lineage (shared by members)
AB_SEED = 20260727
N_SPLITS = 20
RULES = {"bacc": 0, "snap2": 2, "snap3": 3, "snap4": 4}
BAR_EXTEND = 0.00015
BAR_STOP = 0.00005


def fold_indices(y: NDArray[np.int64], seed: int, fold: int) -> tuple[NDArray, NDArray]:
    """Deployed fold split: StratifiedKFold(5, shuffle, random_state=seed), 1-indexed fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        if i == fold:
            return tr, va
    raise ValueError(fold)


def half_split(y_val: NDArray[np.int64], rep: int) -> tuple[NDArray, NDArray]:
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=AB_SEED + rep)
    a, b = next(iter(skf.split(np.zeros(len(y_val)), y_val)))
    return a, b


def bacc(y: NDArray[np.int64], probs: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y, np.argmax(probs, axis=1)))


def capture_member(
    X_all: pd.DataFrame,
    cat_names: list[str],
    y: NDArray[np.int64],
    tr_idx: NDArray[np.int64],
    va_idx: NDArray[np.int64],
    member_seed: int,
) -> NDArray[np.float64]:
    """Fit one member and return its per-epoch val probabilities, shape (E, n_val, 3)."""
    import torch

    import data_science_stuff.kaggle.models.realmlp as rm
    from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

    captured: list[NDArray[np.float64]] = []
    orig_bas = rm.balanced_accuracy_score

    def spy(y_true: Any, y_pred: Any, **kw: Any) -> Any:  # noqa: ANN401
        frame = sys._getframe(1)
        vp = frame.f_locals.get("val_probs")
        if vp is not None:
            captured.append(np.asarray(vp, dtype=np.float64).copy())
        return orig_bas(y_true, y_pred, **kw)

    fold_seed = member_seed + FOLD * 100  # deployed convention
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)  # noqa: NPY002
    model = RealMLP_TD_Classifier(
        train_bs=512,
        fused_optimizer=True,
        allow_amp=True,
        verbosity=1,
        epoch_select="bacc",
        snapshot_k=0,
        random_state=fold_seed,
    )
    rm.balanced_accuracy_score = spy
    try:
        model.fit(
            X_all.iloc[tr_idx],
            y[tr_idx],
            X_all.iloc[va_idx],
            y[va_idx],
            cat_col_names=cat_names,
        )
    finally:
        rm.balanced_accuracy_score = orig_bas
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(captured)


def main() -> None:
    ds = load_dataset()
    y = ds.y
    tr_idx, va_idx = fold_indices(y, SPLIT_SEED_BASE, FOLD)
    y_va = y[va_idx]

    from train_mlp import NNData
    from zoo_common import te_block_for_fold

    nd = NNData(ds)
    cat_names = [f"cat_{c}" for c in nd.cat_cols]
    miss_names = [f"miss_{i}" for i in range(nd.miss_tr.shape[1])]
    te_tr, _te, _n = te_block_for_fold(ds, tr_idx, SPLIT_SEED_BASE, FOLD, "u")
    t13_tr, _t13, _n13 = te13_block_for_fold(ds, tr_idx, SPLIT_SEED_BASE, FOLD, "u")
    te_tr = np.concatenate([te_tr, t13_tr], axis=1)

    num_all, miss_all, cat_all = nd.Xnum_tr, nd.miss_tr, nd.Xcat_tr
    num_f = np.where(np.isnan(num_all), np.nanmedian(num_all[tr_idx], axis=0), num_all)
    te_f = np.where(np.isnan(te_tr), np.nanmedian(te_tr[tr_idx], axis=0), te_tr)
    X_all = pd.DataFrame(
        {
            **{n: cat_all[:, i] for i, n in enumerate(cat_names)},
            **{n: num_f[:, i] for i, n in enumerate(nd.num_cols)},
            **{n: miss_all[:, i] for i, n in enumerate(miss_names)},
            **{f"te_{i}": te_f[:, i] for i in range(te_f.shape[1])},
        }
    )

    lines = [
        "=" * 78,
        "probe_snap_breadth — does snapshot_k=4 survive BREADTH AVERAGING?",
        f"fold {FOLD}, u surface, members {MEMBER_SEEDS} (model seed only; shared split+TE)",
        f"honest A/B: {N_SPLITS} stratified halvings, select on A / score on B (both ways)",
        f"PRE-REG BARS on snap4-bacc at M=4:  >=+{BAR_EXTEND:.5f} escalate / "
        f"<=+{BAR_STOP:.5f} stop / else inconclusive",
        "=" * 78,
        "",
        f"n_train={len(tr_idx)}  n_val={len(va_idx)}  recipe block {te_tr.shape}",
    ]
    print("\n".join(lines))

    cache = RESULTS_DIR / "cache" / "snap_breadth"
    cache.mkdir(parents=True, exist_ok=True)
    members: list[NDArray[np.float64]] = []
    for s in MEMBER_SEEDS:
        t0 = time.time()
        cp = cache / f"epochs_f{FOLD}_m{s}.npy"
        if cp.exists():
            P = np.load(cp)
            how = "from cache"
        else:
            P = capture_member(X_all, cat_names, y, tr_idx, va_idx, s)
            np.save(cp, P)
            how = f"captured in {time.time() - t0:.0f}s"
        members.append(P)
        msg = (
            f"member s{s}: {len(P)} epochs {how}  "
            f"full-val BA per epoch {np.round([bacc(y_va, P[e]) for e in range(len(P))], 6).tolist()}"
        )
        lines.append(msg)
        print(msg)

    # Selection depends only on (member, selection half, rule) — never on M. Precompute
    # the chosen epoch INDICES once (cheap, tiny) and reuse them at every member count,
    # so extending to M=8 costs an argmax pass, not a re-selection pass.
    splits = [half_split(y_va, rep) for rep in range(N_SPLITS)]
    sel_idx: dict[tuple[int, int, int, str], NDArray[np.int64]] = {}
    for mi, P in enumerate(members):
        for rep, (a, b) in enumerate(splits):
            for side, (sel, _sco) in enumerate(((a, b), (b, a))):
                sc = np.asarray([bacc(y_va[sel], P[e][sel]) for e in range(len(P))])
                order = np.argsort(-sc)
                for rname, k in RULES.items():
                    sel_idx[(mi, rep, side, rname)] = order[: max(k, 1)]

    lines += ["", "honest deltas vs bacc-selection (mean over 2*N_SPLITS scored halves):"]
    traj: dict[str, list[float]] = {r: [] for r in RULES}
    for m in range(1, len(members) + 1):
        acc: dict[str, list[float]] = {r: [] for r in RULES}
        for rep, (a, b) in enumerate(splits):
            for side, (_sel, sco) in enumerate(((a, b), (b, a))):
                for rname in RULES:
                    probs = np.mean(
                        [
                            members[mi][sel_idx[(mi, rep, side, rname)]][:, sco].mean(axis=0)
                            for mi in range(m)
                        ],
                        axis=0,
                    )
                    acc[rname].append(bacc(y_va[sco], probs))
        base = np.asarray(acc["bacc"])
        row = [f"  M={m}  bacc {base.mean():.6f}"]
        for rname in RULES:
            if rname == "bacc":
                continue
            d = np.asarray(acc[rname]) - base
            traj[rname].append(float(d.mean()))
            row.append(
                f"{rname} {d.mean():+.6f} (se {d.std(ddof=1) / np.sqrt(len(d)):.6f}, "
                f"{int((d > 0).sum())}/{len(d)})"
            )
        line = "   ".join(row)
        lines.append(line)
        print(line)

    d4 = traj["snap4"][-1]
    lines += ["", f"HEADLINE  snap4 - bacc at M={len(members)} = {d4:+.6f}"]
    lines.append(f"          (M=1 reference: {traj['snap4'][0]:+.6f}; "
                 f"Day-20 honest single-member read was +0.000325)")
    if d4 >= BAR_EXTEND:
        lines.append("VERDICT: ESCALATE — the advantage survives averaging at the bar.")
    elif d4 <= BAR_STOP:
        lines.append(
            "VERDICT: STOP — snapshot averaging and breadth averaging substitute; "
            "#2 unchanged."
        )
    else:
        lines.append("VERDICT: INCONCLUSIVE — between the bars; no swap. #2 unchanged.")
    lines.append(
        "NOTE: #1 stays lrstack_u4_r regardless. Any #2 successor still needs a "
        "submission slot + finals_audit + byte-verify to be clickable."
    )

    out = "\n".join(lines)
    print("\n".join(lines[-6:]))
    suffix = "" if len(members) == 4 else f"_m{len(members)}"
    (RESULTS_DIR / f"probe_snap_breadth{suffix}.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
