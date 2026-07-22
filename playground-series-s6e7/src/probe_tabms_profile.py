"""tabms_te13 breadth as a #2-menu hedge candidate (Day-18 queued item 2).

Veto #12 killed tabms_te13 as a STACK leg; its PLAIN-submission hedge value —
the deployment disagreement profile vs the u-family — was never measured.
This builds the seed-averaged breadth (fresh tag, finals lineage untouched),
fingerprints its test-side disagreements vs every finals-menu candidate
(flip count, AR<->UNH share, missing-driver share — the Day-2 forensics
axes), and refreshes the finals-pairs E[max] MC with it added.

Verdict rule (pre-registered): tabms enters the #2 menu only if some pair
containing it beats the current E[max] ties beyond MC noise (~1e-4) OR its
disagreement profile is structurally distinct (complete-driver flips, not
missing-driver churn) AND solo E[priv] is within the menu band. Otherwise:
measured-out as a hedge, recorded, done.

Run: python src/probe_tabms_profile.py -> results/probe_tabms_profile_day18.txt
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from probe_emax_portfolio import champion_oof_labels, corrected, labels_of, mc_cells
from train_common import (
    CLASSES,
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    TARGET,
    load_dataset,
    robust_decision_weights,
)

from data_science_stuff.kaggle_utils import weighted_predict

SEEDS = [42, 7, 123, 2024, 2025, 31]
NAME = "tabms_te13_u_breadth"
KEY_DRIVERS = ["stress_level", "physical_activity_level", "sleep_duration"]
MENU = [
    "lrstack_u4_r",
    "champion_v1",
    "lrstack6b_r",
    "realmlp_te13_u_breadth",
    "hgbc_te3_u_breadth",
    "lgbm_te3_u_breadth",
]


def main() -> None:
    ds = load_dataset()
    lines: list[str] = []

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    # ---- build the breadth (day13_overnight_u5 pattern, fresh tag) ----
    oofs, tests, used = [], [], []
    for s in SEEDS:
        op = RESULTS_DIR / f"oof_tabms_te13_u_s{s}.npy"
        tp = RESULTS_DIR / f"test_tabms_te13_u_s{s}.npy"
        if op.exists() and tp.exists():
            oofs.append(np.load(op))
            tests.append(np.load(tp))
            used.append(s)
    assert len(used) >= 4, f"only {used} present"
    oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
    np.save(RESULTS_DIR / f"oof_{NAME}.npy", oof)
    np.save(RESULTS_DIR / f"test_{NAME}.npy", test)
    dw = robust_decision_weights(ds.y, oof)
    with (RESULTS_DIR / f"decision_weights_{NAME}.json").open("w") as fh:
        json.dump(dict(zip(CLASSES, dw.tolist())), fh, indent=2)
    wt = balanced_accuracy_score(ds.y, weighted_predict(oof, dw))
    say(f"{NAME}: {len(used)} seeds {used} -> OOF argmax "
        f"{balanced_accuracy_score(ds.y, oof.argmax(1)):.5f}  wt {wt:.5f}")
    test_lab = weighted_predict(test, dw)
    pd.DataFrame({"id": ds.test_ids, TARGET: np.array(CLASSES)[test_lab]}).to_csv(
        SUBMISSIONS_DIR / f"{NAME}.csv", index=False
    )

    # ---- deployment disagreement fingerprint vs the finals menu ----
    md_test = ds.test[KEY_DRIVERS].isna().any(axis=1).to_numpy()
    say(f"\ndisagreement profile (test side, {len(ds.test_ids)} rows; "
        f"missing-driver base {md_test.mean():.3f})")
    say(f"{'candidate':<26} {'flips':>7} {'%':>7} {'AR<->UNH':>9} {'md-share':>9}")
    ar, unh = CLASSES.index("at-risk"), CLASSES.index("unhealthy")
    for cand in MENU:
        csv = SUBMISSIONS_DIR / f"{cand}.csv"
        if csv.exists():
            sub = pd.read_csv(csv)
            assert (sub["id"].to_numpy() == ds.test_ids).all(), f"{cand} row order"
            c_lab = np.array([CLASSES.index(v) for v in sub[TARGET]])
        else:  # never-submitted candidate: compose from arrays (deployed recipe)
            c_lab = labels_of(corrected(cand, "test"))
        d = test_lab != c_lab
        swap = ((test_lab == ar) & (c_lab == unh)) | ((test_lab == unh) & (c_lab == ar))
        say(f"{cand:<26} {int(d.sum()):>7} {100 * d.mean():>6.2f}% "
            f"{swap[d].mean() if d.any() else 0:>9.3f} "
            f"{md_test[d].mean() if d.any() else 0:>9.3f}")

    # ---- finals-pairs E[max] MC refresh with tabms added ----
    y = ds.y
    cands = ["champion_v1", "lrstack6b_r", "lrstack_u4_r",
             "hgbc_te3_u_breadth", "realmlp_te13_u_breadth",
             "lgbm_te3_u_breadth", NAME]
    labs = {
        c: champion_oof_labels(y) if c == "champion_v1"
        else labels_of(corrected(c, "oof"))
        for c in cands
    }
    solo: dict[str, float] = {}
    rows = []
    for a, b in combinations(cands, 2):
        a_p, b_p, _, _ = mc_cells(y, labs[a], labs[b])
        emax = float(np.maximum(a_p, b_p).mean())
        solo[a], solo[b] = float(a_p.mean()), float(b_p.mean())
        rows.append((emax, a, b, float((a_p > b_p).mean())))
    say(f"\n{'candidate':<26} {'E[priv]':>9}")
    for c in cands:
        say(f"  {c:<24} {solo[c]:>9.5f}")
    say(f"\n{'pair':<50} {'E[max]':>9} {'P(first wins)':>14}")
    for emax, a, b, pa in sorted(rows, reverse=True):
        say(f"  {a + ' + ' + b:<48} {emax:>9.5f} {pa:>14.3f}")
    tab_rows = [r for r in rows if NAME in (r[1], r[2])]
    say(f"\nbest tabms pair E[max] {max(tab_rows)[0]:.5f} vs "
        f"best overall {max(rows)[0]:.5f}")
    (RESULTS_DIR / "probe_tabms_profile_day18.txt").write_text("\n".join(lines) + "\n")
    print(f"report -> {RESULTS_DIR / 'probe_tabms_profile_day18.txt'}", flush=True)


if __name__ == "__main__":
    main()
