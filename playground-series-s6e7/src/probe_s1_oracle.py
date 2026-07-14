"""S1 pre-check (Day-11): does the error-placement penalty have a TARGET?

S1 (notes/ideas_day11.md) would train a leg with a penalty that forbids parking
diversity in the info-deleted MCAR region — forcing decorrelation from the core
on complete-driver AND test-like rows (where information EXISTS). It can only pay
if, on that subset, some base of its class is RIGHT where the core is WRONG.

This upgrades the Day-3 headroom oracle (which covered only the old GBDT trio) to
the FULL current diverse-base inventory — the advisor's decisive pre-check. If the
perfect per-row selector over {core + every diverse base} gains > ~0.001 balanced
accuracy on complete-driver ∩ test-like rows, S1 has a real target (build it). If
it lands at ~0 (as Day-3's +0.0002 predicts), no base of S1's class is right where
the core is wrong there — S1 has nothing to decorrelate toward; skip the ½-day GPU.

Plain (unrepaired) surface: the u-family bases and adv/region masks all live there.
Run: python src/probe_s1_oracle.py -> results/probe_s1_oracle.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR, load_dataset

from data_science_stuff.kaggle_utils import weighted_predict

KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
CORE_KEY = "lrstack_u3_r"  # the deployed u-family stack = the reference "core"
# every current diverse base of S1's class (te13-input NN + all diverse families)
CANDIDATE_KEYS = [
    "lrstack_u3_r", "lrstack_u2_r", "lrstack_u_r",
    "realmlp_te13_u_breadth", "realmlp_u_breadth", "realmlp_te13_u_s42",
    "hgbc_te3_u_breadth", "lgbm_te3_u_breadth", "xgb_te3_u_s42",
    "ftt_te13_u_s42", "ftt_te13_u_s7", "ftt_te13_u_s123",
    "tabms_r_s42", "cascade_r_s42", "distill_r_s42",
    "hgbc_te3ncw_u_s42", "hgbc_te3knn_u_s42",
    "mlp", "mlp_la", "realmlp_rule_r_breadth",
    "lgbm_bagw", "hgbc_bagw", "xgboost_bagw", "catboost_bagw",
]


def weighted_pred(key: str, y_len: int):  # noqa: ANN201
    oof = RESULTS_DIR / f"oof_{key}.npy"
    dwj = RESULTS_DIR / f"decision_weights_{key}.json"
    if not oof.exists() or not dwj.exists():
        return None
    arr = np.load(oof)
    if arr.shape != (y_len, 3):
        return None
    d = json.load(open(dwj))
    dw = np.array(list(d["decision"].values())) if "decision" in d else np.array(
        [d[c] for c in ["at-risk", "fit", "unhealthy"]]
    )
    return weighted_predict(arr, dw)


def main() -> None:
    ds = load_dataset()
    y = ds.y
    missing = np.zeros(len(y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    testlike = adv >= np.quantile(adv, 0.70)
    subset = (~missing) & testlike  # complete-driver AND test-like

    preds = {}
    for k in CANDIDATE_KEYS:
        p = weighted_pred(k, len(y))
        if p is not None:
            preds[k] = p
    core = preds[CORE_KEY]

    lines = [
        f"S1 pre-check — perfect-selector oracle on complete-driver ∩ test-like",
        f"core = {CORE_KEY};  bases loaded ({len(preds)}): {sorted(preds)}",
        f"subset: {subset.sum():,} rows ({subset.mean():.1%} of train); "
        f"complete-driver {(~missing).mean():.1%}, test-like {testlike.mean():.1%}",
        "",
    ]

    ys, cs = y[subset], core[subset]
    core_bacc = bacc(ys, cs)
    # perfect per-row selector over ALL loaded bases (incl. core): correct-if-any
    correct_any = np.zeros(subset.sum(), bool)
    for p in preds.values():
        correct_any |= (p[subset] == ys)
    oracle_pred = np.where(correct_any, ys, cs)  # else keep core's (wrong) label
    oracle_bacc = bacc(ys, oracle_pred)

    lines.append(f"  core bacc (subset)          {core_bacc:.5f}")
    lines.append(f"  perfect-selector oracle     {oracle_bacc:.5f}")
    lines.append(f"  ORACLE GAIN                 {oracle_bacc - core_bacc:+.5f}")
    lines.append("")
    # per-base: solo bacc on subset + rows it fixes that the core misses
    core_wrong = cs != ys
    lines.append(f"  {'base':<28} {'solo bacc':>9} {'fixes core-misses':>18}")
    for k, p in sorted(preds.items(), key=lambda kv: -bacc(ys, kv[1][subset])):
        ps = p[subset]
        fixes = int(((ps == ys) & core_wrong).sum())
        lines.append(f"  {k:<28} {bacc(ys, ps):>9.5f} {fixes:>18,}")
    lines.append("")
    lines.append(
        "VERDICT: oracle gain > ~0.001 → S1 has a target, build it. "
        "~0 → S1 has nothing to decorrelate toward, skip the ½-day GPU."
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_s1_oracle.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
