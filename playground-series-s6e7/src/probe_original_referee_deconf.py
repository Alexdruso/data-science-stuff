"""Decision-layer deconfound of the Day-16 referee board (instrument diagnostic).

The pre-registered board read argmax(w*p) with each candidate's own deployed
decision weights — fit on competition OOF priors, applied to a frame whose
priors are shifted (fit 4.9% vs 5.8%, unhealthy 7.6% vs 8.4%). A per-candidate
weight-vector x prior-shift interaction could manufacture the observed gap
pattern without any probability-quality difference. Two equalized boards:

  argmax  — raw blended probabilities, no decision layer at all
  oracle  — per-candidate weights refit ON the referee labels (the decision
            layer's ceiling on this frame; equalizes the layer optimally)

If the stack still trails materially under BOTH, the flag survives the
deconfound; if the gaps collapse, the board pattern is a decision-layer
artifact and the pre-registered read is NULL-with-caveat.

Run: python src/probe_original_referee_deconf.py
  -> results/probe_original_referee_deconf.txt
Requires the referee run's caches (refprobe_*/f*_ref.npy, refprobe_ustack_sref).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES
from lb_anchor import correct
from probe_original_referee import champion_v1_ref, load_referee
from train_common import N_FOLDS, RESULTS_DIR, load_dataset
from zoo_common import CACHE_DIR

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict


def leg_ref(leg: str) -> NDArray[np.float64]:
    return np.mean(
        [np.load(CACHE_DIR / f"refprobe_{leg}" / f"f{f}_ref.npy")
         for f in range(1, N_FOLDS + 1)],
        axis=0,
    )


def main() -> None:
    ds = load_dataset()
    _X_ref, y_ref = load_referee(ds)

    legs = {m: leg_ref(m) for m in ["lgbm", "xgboost", "catboost", "realmlp_te13"]}
    cands: dict[str, NDArray[np.float64]] = {}
    cands["champion_v1"] = champion_v1_ref(ds.y, legs)  # weighted proba (deployed dw)
    s_ref = np.load(CACHE_DIR / "refprobe_ustack_sref.npy")
    with (RESULTS_DIR / "decision_weights_realmlp_te13_u_breadth.json").open() as f:
        rw_map = json.load(f)
    rw = np.array([rw_map[c] for c in CLASSES])
    # raw (pre-decision-layer) probabilities per candidate
    raw = {
        "champion_v1": cands["champion_v1"],  # NM blend already weighted; see note
        "lrstack_u4(6-leg)": s_ref,
        "realmlp_te13 solo": legs["realmlp_te13"],
    }
    # note: champion's pipeline bakes per-model precorrect INTO the blend, so its
    # "raw" is the blended precorrected proba (pre final-dw only). The final dw
    # was applied by correct() in champion_v1_ref, which preserves argmax of w*p;
    # its oracle row below refits on top, which subsumes/overrides the final dw.
    deployed_pred = {
        "champion_v1": cands["champion_v1"].argmax(axis=1),
        "lrstack_u4(6-leg)": None,  # filled below
        "realmlp_te13 solo": correct(legs["realmlp_te13"], rw).argmax(axis=1),
    }
    from train_common import robust_decision_weights

    soof = np.load(CACHE_DIR / "refprobe_ustack_soof.npy")
    dw_stack = robust_decision_weights(ds.y, soof)
    deployed_pred["lrstack_u4(6-leg)"] = weighted_predict(s_ref, dw_stack)

    lines = ["referee board deconfound — decision-layer variants", ""]
    lines.append(f"{'candidate':22s} {'deployed':>9s} {'argmax':>9s} {'oracle':>9s}")
    for name, p in raw.items():
        ba_dep = balanced_accuracy_score(y_ref, deployed_pred[name])
        ba_arg = balanced_accuracy_score(y_ref, p.argmax(axis=1))
        w_or = tune_decision_weights(y_ref, p)
        ba_or = balanced_accuracy_score(y_ref, weighted_predict(p, w_or))
        lines.append(f"{name:22s} {ba_dep:9.5f} {ba_arg:9.5f} {ba_or:9.5f}")
    report = "\n".join(lines)
    print(report, flush=True)
    (RESULTS_DIR / "probe_original_referee_deconf.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
