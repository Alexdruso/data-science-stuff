"""Where does a non-GBDT base's diversity live — and does it TRANSFER? (diagnose + gate)

Two jobs, one tool:
  1. Diagnose a candidate leg: is its error-fixing concentrated in the
     missing-driver / non-test-like region (the trap that regressed ensemble v2), or does
     it also help the test-like rows the LB is drawn from?
  2. Gate any new base (default `mlp_r`): a base SURVIVES only if it CLEARLY lifts
     adv-weighted OOF AND adds diversity in the TEST-LIKE region. Sub-0.001 wiggles are
     shift-noise (adv_eval / lb_anchor established this) → discard, per the LB-blind protocol.

Day-6 gate fix (2026-07-08):
  - The comparator core is the DEPLOYED repaired 8-seed breadth blend
    (`oof_ensemble_r_breadth.npy`, already precorrected + NM-blended by combine_breadth.py —
    its weights JSON is nested, so it must NOT go through precorrected_blend again).
    Override with S6E7_CORE.
  - Adversarial scores are cached per SURFACE (adv_scores_train_r.npy under S6E7_REPAIR=1) —
    the old cache was silently reused because the key was shape-only. "Test-like" is defined
    on the REPAIRED surface: the repair removed the mask-mechanism shift, so repaired adv
    scores isolate the residual (numeric/gender) shift the deployed stack still faces.
  - Surfaces must MATCH: repaired OOF vs plain-surface region masks/adv scores (or vice
    versa) is confounded — main() hard-fails if the core key and S6E7_REPAIR disagree.
  - The candidate joins the core as a fixed-w mixture (W_GATE=0.20 ≈ a 5th family joining a
    4-family blend, the old gate's marginal-member semantics). NM weights on a near-tie OOF
    hand the new leg an overfit top weight (Day-5 lgbm_dp); equal weight would inflate lift.

Read-only over results/oof_*.npy. Usage: `S6E7_REPAIR=1 python src/diag_mlp_transfer.py [model_key ...]`.
Reuses adv_eval (adversarial scores, precorrected blend, weighted bacc) + train_common weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from adv_eval import adversarial_scores, precorrected_blend, weighted_bacc
from train_common import REPAIR, RESULTS_DIR, load_dataset, robust_decision_weights

KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
CORE_KEY = os.environ.get("S6E7_CORE", "ensemble_r_breadth")
ADV_CACHE = RESULTS_DIR / ("adv_scores_train_r.npy" if REPAIR else "adv_scores_train.npy")
W_GATE = 0.20
W_GRID = (0.10, 0.20, 0.30, 0.50)


def load_adv(ds: object, n_train: int) -> NDArray[np.float64]:
    """Reuse the cached train-row P(test) for THIS surface, else recompute (and cache)."""
    if ADV_CACHE.exists():
        adv = np.load(ADV_CACHE)
        if adv.shape == (n_train,):
            print(f"adv scores: loaded cache {ADV_CACHE.name} "
                  f"(surface={'REPAIRED' if REPAIR else 'plain'})")
            return adv
    adv = adversarial_scores(ds)
    np.save(ADV_CACHE, adv)
    return adv


def wbacc(y: NDArray, proba: NDArray) -> NDArray[np.int64]:
    """Decision-weighted argmax prediction using bagged (robust) weights."""
    dw = robust_decision_weights(y, proba)
    return np.asarray(np.argmax(proba * dw, axis=1), dtype=np.int64)


def diagnose(m_key: str, y: NDArray, adv: NDArray, missing: NDArray) -> None:
    # Already precorrected + NM-blended by combine_breadth.py (nested weights JSON —
    # must not pass through precorrected_blend); this IS the deployed pre-decision surface.
    core_blend = np.load(RESULTS_DIR / f"oof_{CORE_KEY}.npy")
    try:
        m_oof = np.load(RESULTS_DIR / f"oof_{m_key}.npy")
    except FileNotFoundError:
        print(f"\n[{m_key}] oof_{m_key}.npy not found — skip (run its trainer first).")
        return
    if ("_r" in m_key) != REPAIR:
        print(f"WARNING: candidate key '{m_key}' looks "
              f"{'plain' if REPAIR else 'repaired'}-surface but the run is "
              f"{'REPAIRED' if REPAIR else 'plain'} — verdict is confounded.")

    testlike = adv >= np.quantile(adv, 0.70)
    print(f"\n{'='*70}\n[{m_key}] transfer diagnosis vs core={CORE_KEY}")
    print(
        f"regions: missing-driver {missing.mean():.1%} of train, "
        f"test-like (top-30% adv) {testlike.mean():.1%}"
    )

    # --- 1. Solo decision-weighted bacc by region -------------------------------------
    core_pred = wbacc(y, core_blend)
    m_pred = wbacc(y, m_oof)
    print(f"\n  solo decision-weighted balanced acc:")
    print(f"    {'subset':<22}{'core':>8}{m_key:>10}{'Δ':>9}")
    for name, mask in [
        ("overall", np.ones(len(y), bool)),
        ("test-like", testlike),
        ("NOT test-like", ~testlike),
        ("missing-driver", missing),
        ("complete-driver", ~missing),
    ]:
        c = balanced_accuracy_score(y[mask], core_pred[mask])
        mm = balanced_accuracy_score(y[mask], m_pred[mask])
        print(f"    {name:<22}{c:>8.4f}{mm:>10.4f}{mm - c:>+9.4f}")

    # --- 2. Where does m FIX the core's errors? How decorrelated is it? ---------------
    core_wrong = core_pred != y
    m_right = m_pred == y
    fixes = core_wrong & m_right  # core wrong, m right
    print(f"\n  of the core's {core_wrong.sum()} errors, {m_key} is right on "
          f"{fixes.sum()} ({fixes.sum()/max(core_wrong.sum(),1):.1%}). Where they live:")
    for name, mask in [("test-like", testlike), ("NOT test-like", ~testlike),
                       ("missing-driver", missing), ("complete-driver", ~missing)]:
        share = fixes[mask].sum() / max(fixes.sum(), 1)
        print(f"    {name:<18}{fixes[mask].sum():>7}  ({share:.1%} of fixes)")
    m_wrong = ~m_right
    for name, mask in [("overall", np.ones(len(y), bool)), ("test-like", testlike)]:
        both_wrong = (m_wrong & core_wrong & mask).sum()
        overlap = both_wrong / max((m_wrong & mask).sum(), 1)
        counter = ((m_wrong & ~core_wrong & mask).sum() / max((m_wrong & mask).sum(), 1))
        print(f"    error-overlap {name:<10} |err(m)∩err(core)|/|err(m)| = {overlap:.1%}"
              f"  (core right where m wrong: {counter:.1%})")

    # --- 3. Marginal ensemble lift (THE GATE) -----------------------------------------
    m_corr = precorrected_blend({m_key: m_oof}, [m_key])
    tl_core = balanced_accuracy_score(y[testlike], core_pred[testlike])
    aw_core = weighted_bacc(y, core_pred, adv)
    print(f"\n  ensemble (core vs (1-w)·core + w·{m_key}) — plain / test-like / advwt bacc:")
    print(f"    {'w':<16}{balanced_accuracy_score(y, core_pred):>9.4f}"
          f"{tl_core:>12.4f}{aw_core:>13.4f}   (core)")
    aw_gate, tl_gate = 0.0, 0.0
    for w in W_GRID:
        ens = (1 - w) * core_blend + w * m_corr
        ens_pred = wbacc(y, ens)
        plain = balanced_accuracy_score(y, ens_pred)
        tl = balanced_accuracy_score(y[testlike], ens_pred[testlike])
        aw = weighted_bacc(y, ens_pred, adv)
        mark = "  ← gate" if w == W_GATE else ""
        print(f"    w={w:<14.2f}{plain:>9.4f}{tl:>12.4f}{aw:>13.4f}{mark}")
        if w == W_GATE:
            aw_gate, tl_gate = aw, tl
    verdict = "SURVIVES" if (aw_gate - aw_core > 0.001 and tl_gate - tl_core > 0) else "veto"
    print(f"\n  GATE (w={W_GATE}): adv-weighted Δ={aw_gate-aw_core:+.4f}  "
          f"test-like Δ={tl_gate-tl_core:+.4f}"
          f"  → {verdict} (need advwt Δ>+0.001 AND test-like Δ>0)")


def main() -> None:
    keys = sys.argv[1:] or ["mlp_r"]
    print(f"surface={'REPAIRED' if REPAIR else 'plain'}  core={CORE_KEY}")
    if ("_r" in CORE_KEY) != REPAIR:
        raise SystemExit(
            f"surface mismatch: core '{CORE_KEY}' vs S6E7_REPAIR={'1' if REPAIR else 'unset'}"
            " — repaired core needs S6E7_REPAIR=1 (region masks + adv scores must be on the"
            " same surface as the OOF arrays)."
        )
    ds = load_dataset()
    y = ds.y
    missing = np.zeros(len(y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    adv = load_adv(ds, len(y))
    for k in keys:
        diagnose(k, y, adv, missing)


if __name__ == "__main__":
    main()
