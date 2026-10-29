"""PR-46 diagnostic: detect val-fold-selection inflation in stack members without labels on test.

    python src/inflation_scan.py --ref LEG [--ours LEG ...]

For each member, corr(member, ref) on OOF vs on test (logits). An honest member's OOF and test
predictions are the same function of the features, so the two correlations match up to the
fold-averaging effect on test. A member whose OOF was selected on its own val labels (epoch
picked by val AUC, etc.) carries fitted label noise on OOF only, which lowers its OOF correlation
with an honest reference. Also prints corr(member residual vs ref, y) on OOF: the label signal
the member holds beyond the reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from stack_public import load_pairs

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, _ = competition_dirs(__file__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--public", default=str(DATA_DIR / "public_oof" / "_pool"))
    p.add_argument("--out", default=str(RESULTS_DIR / "inflation_scan.csv"))
    a = p.parse_args()
    y = build_features(pl.read_csv(DATA_DIR / "train.csv"))[TARGET].to_numpy().astype(float)
    names, O, T = load_pairs([a.ref, *a.ours], [a.public])
    O, T = np.clip(O, -30, 30), np.clip(T, -30, 30)
    r_o, r_t = O[:, 0], T[:, 0]
    rows = []
    for i, n in enumerate(names[1:], 1):
        o, t = O[:, i], T[:, i]
        c_o, c_t = np.corrcoef(o, r_o)[0, 1], np.corrcoef(t, r_t)[0, 1]
        # residual of member on ref (OLS), its correlation with the label residual
        b = np.polyfit(r_o, o, 1)
        res = o - np.polyval(b, r_o)
        rows.append({"member": n, "c_oof": c_o, "c_test": c_t, "d_corr": c_t - c_o,
                     "sd_ratio": t.std() / o.std(), "res_y": np.corrcoef(res, y)[0, 1]})
    df = pl.DataFrame(rows).sort("d_corr", descending=True)
    df.write_csv(a.out)
    with pl.Config(tbl_rows=400, tbl_width_chars=200, fmt_str_lengths=70):
        print(df)


if __name__ == "__main__":
    main()
