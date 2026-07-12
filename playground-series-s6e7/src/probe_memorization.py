"""Row-memorization exact-match transfer probe (Day-10, GF #6).

Question: does the generator memorize original rows verbatim, so an exact
13-column hash-join test->original recovers labels? Day-4's probe_linkage
already found 0 exact matches on the 7-numeric key (which dominates this for
complete-numeric rows); this closes the remaining channel with a measured
artifact: the all-13 string key (the te13 casting — NaN = 'nan' level) joined
for ALL rows, plus a per-column-permuted null and, if any matches exist, the
label-agreement read that would license a post-hoc overwrite (>0.992).

Run: python src/probe_memorization.py   -> results/probe_memorization.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET
from train_common import DATA_DIR, RESULTS_DIR, load_dataset

EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"


def key_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    view = df[cols].astype(str)
    key = view[cols[0]]
    for c in cols[1:]:
        key = key + "|" + view[c]
    return key


def main() -> None:
    ds = load_dataset()
    ext = pd.read_csv(EXT_CSV)
    cols = ds.feature_cols
    assert all(c in ext.columns for c in cols), "external schema mismatch"

    ext_key = key_series(ext, cols)
    ext_labels = ext[TARGET]
    ext_map = pd.Series(ext_labels.to_numpy(), index=ext_key.to_numpy())
    ext_map = ext_map[~ext_map.index.duplicated()]

    lines = [f"13-col string key; external rows {len(ext)}, "
             f"unique keys {ext_key.nunique()}"]
    rng = np.random.default_rng(42)
    for name, frame in (("test", ds.test), ("train", ds.train)):
        key = key_series(frame, cols)
        matched = key.isin(ext_map.index)
        # permuted null: shuffle each column independently -> same marginals,
        # destroyed joint structure
        perm = pd.DataFrame(
            {c: frame[c].sample(frac=1.0, random_state=int(rng.integers(1e9)))
             .to_numpy() for c in cols}
        )
        null_matched = key_series(perm, cols).isin(ext_map.index)
        lines.append(
            f"{name}: exact matches {int(matched.sum())} / {len(frame)} "
            f"(permuted null {int(null_matched.sum())})"
        )
        if name == "train" and matched.any():
            agree = (
                ext_map.loc[key[matched].to_numpy()].to_numpy()
                == np.array(["at-risk", "fit", "unhealthy"])[ds.y[matched.to_numpy()]]
            )
            lines.append(
                f"  train-match label agreement: {agree.mean():.4f} "
                f"(overwrite lever needs >0.992)"
            )

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_memorization.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
