"""S1 diag signature (Day-11): did the error-placement penalty move WHERE the
errors land? The pre-registered read (advisor / notes/ideas_day11.md):

  success = fix-share rises AND %fixes-in-missing-driver FALLS below the universal
  ~90% (every prior NN: mlp_r 90.2%, dann_r 95.2%) for the first time.
  kill = signature stays ≥90% missing-driver → error placement is a hard invariant.

Reference core = lrstack_u3 (u-family). Plain/unrepaired surface masks (drivers +
adversarial top-30%), same as probe_s1_oracle.py.

Run: python src/probe_s1_signature.py <s1_key> [<s1_key> ...]
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
CORE_KEY = "lrstack_u3_r"


def wpred(key: str, n: int):  # noqa: ANN201
    d = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
    dw = np.array(list(d["decision"].values())) if "decision" in d else np.array(
        [d[c] for c in ["at-risk", "fit", "unhealthy"]]
    )
    return weighted_predict(np.load(RESULTS_DIR / f"oof_{key}.npy"), dw)


def main() -> None:
    keys = sys.argv[1:]
    if not keys:
        raise SystemExit("usage: probe_s1_signature.py <s1_key> ...")
    ds = load_dataset()
    y = ds.y
    missing = np.zeros(len(y), bool)
    for d in KEY_DRIVERS:
        missing |= ds.train[d].isna().to_numpy()
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")
    testlike = adv >= np.quantile(adv, 0.70)
    core = wpred(CORE_KEY, len(y))
    core_wrong = core != y

    print(f"core={CORE_KEY}  weighted bacc {bacc(y, core):.4f}  "
          f"(missing-driver {missing.mean():.1%}, test-like {testlike.mean():.1%})")
    print(f"{'cand':<20} {'wbacc':>7} {'fix-share':>9} "
          f"{'%fix miss-drv':>13} {'%fix testlk':>11} {'err-overlap':>11}")
    for k in keys:
        cand = wpred(k, len(y))
        cand_wrong = cand != y
        fixes = core_wrong & (cand == y)
        breaks = (~core_wrong) & cand_wrong
        fix_share = fixes.sum() / max(core_wrong.sum(), 1)
        pct_miss = (fixes & missing).sum() / max(fixes.sum(), 1)
        pct_tl = (fixes & testlike).sum() / max(fixes.sum(), 1)
        overlap = (core_wrong & cand_wrong).sum() / max((core_wrong | cand_wrong).sum(), 1)
        print(f"{k:<20} {bacc(y, cand):>7.4f} {fix_share:>9.1%} "
              f"{pct_miss:>13.1%} {pct_tl:>11.1%} {overlap:>11.1%}"
              f"   (fixes {int(fixes.sum())}, breaks {int(breaks.sum())})")


if __name__ == "__main__":
    main()
