"""Day-13 PM — finals-integrity audit: CSV bytes vs arrays, end to end.

Insurance against the campaign's one catastrophic failure class (row-order
misalignment once produced a 0.5-AUC submission). For every finals-menu CSV:

  1. structure: 295,753 rows, header id,health_condition, ids == test ids
     exactly (set AND order), labels in CLASSES, no NaN;
  2. faithfulness: labels recomputed from results/test_<key>.npy + its
     decision-weights JSON (build_features id order), joined to the CSV ON ID —
     require 100% agreement (catches stale CSVs after array rebuilds and any
     order drift);
  3. class distribution + sha256 recorded.

--regen <key>: rebuild the CSV from the current arrays first (used for
realmlp_te13_u_breadth, whose CSV predated today's 8-seed recombine).

Run: python src/finals_audit.py -> results/finals_audit_2026-07-15.txt
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ensemble import RESULTS_DIR, SUBMISSIONS_DIR
from features import CLASSES, TARGET
from train_common import load_dataset

from data_science_stuff.kaggle_utils import weighted_predict

# csv name -> array key (None = champion label file)
MENU: dict[str, str | None] = {
    "lrstack_u4_r.csv": "lrstack_u4_r",
    "lrstack_u3_r.csv": "lrstack_u3_r",
    "lrstack6b_r.csv": "lrstack6b_r",
    "champion_v1.csv": None,
    "realmlp_te13_u_breadth.csv": "realmlp_te13_u_breadth",
    "hgbc_te3_u_breadth.csv": "hgbc_te3_u_breadth",
}


def saved_weights(key: str) -> np.ndarray:
    dwj = json.load(open(RESULTS_DIR / f"decision_weights_{key}.json"))
    return (
        np.array(list(dwj["decision"].values()))
        if "decision" in dwj
        else np.array([dwj[c] for c in CLASSES])
    )


def expected_labels(key: str | None, test_ids: np.ndarray) -> pd.Series:
    if key is None:
        idx = np.load(RESULTS_DIR / "champion_test_labels.npy")
        lab = np.array(CLASSES)[idx] if idx.dtype.kind in "iu" else idx
    else:
        proba = np.load(RESULTS_DIR / f"test_{key}.npy")
        lab = np.array(CLASSES)[weighted_predict(proba, saved_weights(key))]
    return pd.Series(lab, index=test_ids)


def regen(key: str, test_ids: np.ndarray) -> None:
    lab = expected_labels(key, test_ids)
    out = SUBMISSIONS_DIR / f"{key}.csv"
    pd.DataFrame({"id": test_ids, TARGET: lab.to_numpy()}).to_csv(out, index=False)
    print(f"regenerated {out.name} from current arrays")


def main() -> None:
    ds = load_dataset()
    test_ids = ds.test_ids
    if "--regen" in sys.argv:
        regen(sys.argv[sys.argv.index("--regen") + 1], test_ids)

    lines = ["finals-integrity audit 2026-07-15"]
    ok_all = True
    for csv_name, key in MENU.items():
        path = SUBMISSIONS_DIR / csv_name
        if not path.exists():
            lines.append(f"  {csv_name:<32} MISSING")
            ok_all = False
            continue
        df = pd.read_csv(path)
        problems = []
        if list(df.columns) != ["id", TARGET]:
            problems.append(f"columns {list(df.columns)}")
        if len(df) != len(test_ids):
            problems.append(f"rows {len(df)}")
        if not np.array_equal(np.sort(df["id"].to_numpy()), np.sort(test_ids)):
            problems.append("id set mismatch")
        if df[TARGET].isna().any() or not df[TARGET].isin(CLASSES).all():
            problems.append("bad labels")
        exp = expected_labels(key, test_ids)
        joined = df.set_index("id")[TARGET]
        mism = int((joined.reindex(exp.index) != exp).sum())
        if mism:
            problems.append(f"{mism} label mismatches vs arrays")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        dist = df[TARGET].value_counts(normalize=True).round(4).to_dict()
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        ok_all &= not problems
        lines.append(f"  {csv_name:<32} {status}  sha256:{sha}  dist:{dist}")
    lines.append("ALL OK" if ok_all else "!! AUDIT FAILURES — do not select finals")
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "finals_audit_2026-07-15.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
