"""Transferable-headroom bound for PS S6E7 — is any current-base recombination worth it?

A perfect per-row SELECTOR over {champion core, mlp_la, lgbm_ext} is the ceiling of what any
stacker/gating of these bases could achieve. The overall oracle over-counts, because most of the
selectable gain lives in the missing-driver / non-test-like region that does NOT transfer to the
LB (measured three times today). So the decisive number is the oracle gain restricted to
complete-driver ∩ test-like rows — the only region where a gain both EXISTS and TRANSFERS.

If that restricted gain is <0.001 → current-base recombination is *measured*-exhausted (a BOUND,
not a "ceiling"): stop adding bases. If >0.001 → a region-restricted selector is a real lever.

Read-only. Reuses diag_mlp_transfer (adv/region/weights) + adv_eval (core blend).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from adv_eval import GBDTS, precorrected_blend
from diag_mlp_transfer import KEY_DRIVERS, load_adv, wbacc
from train_common import RESULTS_DIR, load_dataset

BASES = ["mlp_la", "lgbm_ext"]  # candidate diverse bases vs the champion core


def oracle_delta(y, core_pred, base_preds, allowed):
    """Balanced-acc gain of a perfect selector that may override core only inside `allowed`."""
    oracle = core_pred.copy()
    right_any = np.zeros(len(y), bool)
    for p in base_preds:
        right_any |= p == y
    fixable = allowed & (core_pred != y) & right_any
    oracle[fixable] = y[fixable]
    return (
        balanced_accuracy_score(y, oracle) - balanced_accuracy_score(y, core_pred),
        int(fixable.sum()),
    )


def main() -> None:
    ds = load_dataset()
    y = ds.y
    adv = load_adv(ds, len(y))
    testlike = adv >= np.quantile(adv, 0.70)
    missing = np.zeros(len(y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    complete = ~missing

    core_oof = {m: np.load(RESULTS_DIR / f"oof_{m}.npy") for m in GBDTS}
    core_pred = wbacc(y, precorrected_blend(core_oof, GBDTS))
    base_preds = [wbacc(y, np.load(RESULTS_DIR / f"oof_{b}.npy")) for b in BASES]

    print(f"perfect-selector oracle over core + {BASES}")
    print(f"  core balanced_acc: {balanced_accuracy_score(y, core_pred):.4f}\n")
    print(f"  {'allowed region':<28}{'ΔbACC':>9}{'fixable rows':>14}")
    for name, allowed in [
        ("all rows (upper bound)", np.ones(len(y), bool)),
        ("test-like", testlike),
        ("complete-driver", complete),
        ("complete ∩ test-like  ⟵", complete & testlike),
    ]:
        d, n = oracle_delta(y, core_pred, base_preds, allowed)
        print(f"  {name:<28}{d:>+9.4f}{n:>14}")
    print("\n  → the 'complete ∩ test-like' row is the transferable headroom bound.")
    print("    <0.001 ⇒ current-base recombination is measured-exhausted (a bound, not a ceiling).")


if __name__ == "__main__":
    main()
