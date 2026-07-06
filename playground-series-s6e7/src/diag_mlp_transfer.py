"""Where does a non-GBDT base's diversity live — and does it TRANSFER? (diagnose + gate)

Two jobs, one tool:
  1. Diagnose the existing `mlp`: is its error-fixing concentrated in the
     missing-driver / non-test-like region (the trap that regressed ensemble v2), or does
     it also help the test-like rows the LB is drawn from?
  2. Gate any new base (default `mlp_la`): a base SURVIVES only if it CLEARLY lifts
     adv-weighted OOF AND adds diversity in the TEST-LIKE region. Sub-0.001 wiggles are
     shift-noise (adv_eval / lb_anchor established this) → discard, per the LB-blind protocol.

Read-only over results/oof_*.npy. Usage: `python src/diag_mlp_transfer.py [model_key ...]`.
Reuses adv_eval (adversarial scores, precorrected blend, weighted bacc) + train_common weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from adv_eval import GBDTS, adversarial_scores, precorrected_blend, weighted_bacc
from train_common import RESULTS_DIR, load_dataset, robust_decision_weights

KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
ADV_CACHE = RESULTS_DIR / "adv_scores_train.npy"


def load_adv(ds: object, n_train: int) -> NDArray[np.float64]:
    """Reuse the cached train-row P(test) if length matches, else recompute (and cache)."""
    if ADV_CACHE.exists():
        adv = np.load(ADV_CACHE)
        if adv.shape == (n_train,):
            print(f"adv scores: loaded cache {ADV_CACHE.name}")
            return adv
    adv = adversarial_scores(ds)
    np.save(ADV_CACHE, adv)
    return adv


def wbacc(y: NDArray, proba: NDArray) -> NDArray[np.int64]:
    """Decision-weighted argmax prediction using bagged (robust) weights."""
    dw = robust_decision_weights(y, proba)
    return np.asarray(np.argmax(proba * dw, axis=1), dtype=np.int64)


def diagnose(m_key: str, ds: object, y: NDArray, adv: NDArray, missing: NDArray) -> None:
    core_oof = {m: np.load(RESULTS_DIR / f"oof_{m}.npy") for m in GBDTS}
    try:
        m_oof = np.load(RESULTS_DIR / f"oof_{m_key}.npy")
    except FileNotFoundError:
        print(f"\n[{m_key}] oof_{m_key}.npy not found — skip (run its trainer first).")
        return

    testlike = adv >= np.quantile(adv, 0.70)
    print(f"\n{'='*70}\n[{m_key}] transfer diagnosis vs 3-GBDT core")
    print(
        f"regions: missing-driver {missing.mean():.1%} of train, "
        f"test-like (top-30% adv) {testlike.mean():.1%}"
    )

    # --- 1. Solo decision-weighted bacc by region -------------------------------------
    core_blend = precorrected_blend(core_oof, GBDTS)
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

    # --- 2. Where does m FIX the core's errors? ---------------------------------------
    core_wrong = core_pred != y
    m_right = m_pred == y
    fixes = core_wrong & m_right  # core wrong, m right
    print(f"\n  of the core's {core_wrong.sum()} errors, {m_key} is right on "
          f"{fixes.sum()} ({fixes.sum()/max(core_wrong.sum(),1):.1%}). Where they live:")
    for name, mask in [("test-like", testlike), ("NOT test-like", ~testlike),
                       ("missing-driver", missing), ("complete-driver", ~missing)]:
        share = fixes[mask].sum() / max(fixes.sum(), 1)
        print(f"    {name:<18}{fixes[mask].sum():>7}  ({share:.1%} of fixes)")

    # --- 3. Marginal ensemble lift (THE GATE) -----------------------------------------
    both = {**core_oof, m_key: m_oof}
    ens = precorrected_blend(both, [*GBDTS, m_key])
    ens_pred = wbacc(y, ens)
    print(f"\n  ensemble (core vs core+{m_key}) — plain / test-like / adv-weighted bacc:")
    for tag, pred in [("core", core_pred), (f"core+{m_key}", ens_pred)]:
        plain = balanced_accuracy_score(y, pred)
        tl = balanced_accuracy_score(y[testlike], pred[testlike])
        aw = weighted_bacc(y, pred, adv)
        print(f"    {tag:<16}{plain:>9.4f}{tl:>12.4f}{aw:>13.4f}")
    aw_core = weighted_bacc(y, core_pred, adv)
    aw_ens = weighted_bacc(y, ens_pred, adv)
    tl_core = balanced_accuracy_score(y[testlike], core_pred[testlike])
    tl_ens = balanced_accuracy_score(y[testlike], ens_pred[testlike])
    verdict = "SURVIVES" if (aw_ens - aw_core > 0.001 and tl_ens - tl_core > 0) else "veto"
    print(f"\n  GATE: adv-weighted Δ={aw_ens-aw_core:+.4f}  test-like Δ={tl_ens-tl_core:+.4f}"
          f"  → {verdict} (need advwt Δ>+0.001 AND test-like Δ>0)")


def main() -> None:
    keys = sys.argv[1:] or ["mlp"]
    ds = load_dataset()
    y = ds.y
    missing = np.zeros(len(y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    adv = load_adv(ds, len(y))
    for k in keys:
        diagnose(k, ds, y, adv, missing)


if __name__ == "__main__":
    main()
