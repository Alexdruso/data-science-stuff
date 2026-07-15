"""Day-13 Lane C — offline δ_r triangulation from the untouched-row instrument.

δ_r (Day-10 PM decision-theorist framing) = how much the repaired lineage's OOF
understates its deployment score relative to the u lineage (knife-edge for the
two-surface finals pair: δ_r* ≈ 0.0012; public paired read gave ≈0.0009,
band [0.0004, 0.0013]).

New offline datum: partition val rows by whether the m100 remask TOUCHED them
(flipped ≥1 cell in the 4 shifted columns). On UNTOUCHED rows (~87%) the r- and
u-lineage val inputs are IDENTICAL (this is a superset of read_lineage's
complete-in-4: it adds originally-NaN-in-4 rows the remask left alone), so the
score gap there is a true model-quality gap, no surface artifact. The OOF gap
on TOUCHED rows mixes model quality with the harder-inputs artifact — the
surface-induced share of the overall OOF gap is what δ_r measures.

Decomposition (saved deployed weights, deployment-faithful):
  gap_all = share_U·gap_U + share_T·gap_T(OOF surface)
  δ_r estimate ≈ gap_all − gap_deploy, bracketing deployment gap on touched
  rows between gap_U (r at its untouched par) and gap_T (no artifact at all).

Run: python src/probe_delta_r.py   (no S6E7_REPAIR)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES

from data_science_stuff.kaggle_utils import weighted_predict

assert os.environ.get("S6E7_REPAIR", "") in ("", "0"), "run without S6E7_REPAIR"
from train_common import (  # noqa: E402
    MECHANISM_SHIFTED,
    RESULTS_DIR,
    _uniform_remask,
    load_dataset,
)

U_KEY = "lrstack_u3_r"
R_KEY = "lrstack_rte13_r"


def saved_weights(key: str) -> np.ndarray:
    dwj = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
    return (
        np.array(list(dwj["decision"].values()))
        if "decision" in dwj
        else np.array([dwj[c] for c in CLASSES])
    )


def main() -> None:
    ds = load_dataset()
    y = ds.y
    os.environ["S6E7_REPAIR_MULT"] = "1.0"
    remasked = _uniform_remask(ds.train, ds.test)
    cols = [c for c in MECHANISM_SHIFTED if c in ds.train.columns]
    untouched = np.ones(len(y), dtype=bool)
    for c in cols:
        untouched &= (
            ds.train[c].isna().to_numpy() == remasked[c].isna().to_numpy()
        )
    print(
        f"untouched rows (remask flipped nothing in the 4 cols): "
        f"{untouched.sum():,} ({untouched.mean():.1%}); touched {(~untouched).sum():,}"
    )

    scores: dict[str, dict[str, float]] = {}
    for key in (U_KEY, R_KEY):
        oof = np.load(RESULTS_DIR / f"oof_{key}.npy")
        pred = weighted_predict(oof, saved_weights(key))
        scores[key] = {
            "all": bacc(y, pred),
            "untouched": bacc(y[untouched], pred[untouched]),
            "touched": bacc(y[~untouched], pred[~untouched]),
        }
        s = scores[key]
        print(
            f"{key:<18} all={s['all']:.5f} untouched={s['untouched']:.5f} "
            f"touched={s['touched']:.5f}"
        )

    gap = {
        k: scores[U_KEY][k] - scores[R_KEY][k] for k in ("all", "untouched", "touched")
    }
    print(
        f"\nu3 − rte13_r gaps: all={gap['all']:+.5f} "
        f"untouched={gap['untouched']:+.5f} touched={gap['touched']:+.5f}"
    )
    # bacc is not row-additive across subsets, so bracket in gap units directly:
    # deployment gap ∈ [gap_untouched (r reaches its untouched par on touched
    # rows too), gap_all (surface artifact is entirely real)]
    # δ_r = OOF gap − deployment gap
    lo, hi = gap["all"] - gap["all"], gap["all"] - gap["untouched"]
    print(
        f"δ_r offline bracket: [{lo:+.5f}, {hi:+.5f}] "
        f"(hi = surface-induced share if r is at untouched-par everywhere; "
        f"knife-edge δ_r* ≈ +0.0012; public paired read ≈ +0.0009)"
    )


if __name__ == "__main__":
    main()
