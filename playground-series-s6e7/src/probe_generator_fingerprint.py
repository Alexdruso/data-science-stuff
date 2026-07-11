"""LH-6 generator fingerprinting (Day-9): is the synthetic generator a
per-cell conditional-marginal sampler?

(a) Value-vocabulary containment: synthetic generators trained per-column emit
    values from the original's vocabulary; fraction of train's unique numeric
    values contained in the 50k original + frequency correlation on shared
    values.
(b) Within-class (and within-rule-cell) correlation shrinkage: if the
    generator samples features INDEPENDENTLY given the class/cell, train's
    within-cell numeric correlations collapse toward 0 while the original
    keeps its structure.

If (b) confirms independence-given-cell, P(y | x_observed) is analytically
computable by per-cell naive Bayes over the exact rule table — a generative
model family that is correct by construction on the missing-driver region
(prob-purist #4's arbitrage becomes concrete).

Report -> results/probe_generator_fingerprint.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS
from rule_features import sleep_bucket
from train_common import DATA_DIR, RESULTS_DIR

EXT = DATA_DIR / "external" / "student_health_dataset_50k.csv"
TARGET_EXT = "health_condition"


def containment(train: pd.DataFrame, ext: pd.DataFrame, out: list[str]) -> None:
    out.append("=" * 70)
    out.append("LH-6a  value-vocabulary containment (train uniques within original 50k)")
    out.append("=" * 70)
    for c in NUM_COLS:
        tv = train[c].dropna()
        ev = ext[c].dropna()
        t_set, e_set = set(tv.unique()), set(ev.unique())
        cont = len(t_set & e_set) / max(len(t_set), 1)
        # frequency correlation on shared values
        shared = sorted(t_set & e_set)
        if len(shared) >= 10:
            tf = tv.value_counts(normalize=True).reindex(shared).to_numpy()
            ef = ev.value_counts(normalize=True).reindex(shared).to_numpy()
            fc = float(np.corrcoef(tf, ef)[0, 1])
        else:
            fc = float("nan")
        out.append(
            f"  {c:<22} train uniq {len(t_set):>6}  orig uniq {len(e_set):>6}  "
            f"containment {cont:.3f}  freq-corr {fc:+.3f}"
        )


def corr_offdiag(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    m = df[NUM_COLS].corr().to_numpy()
    off = m[np.triu_indices_from(m, k=1)]
    return float(np.nanmean(np.abs(off))), off


def shrinkage(train: pd.DataFrame, ext: pd.DataFrame, out: list[str]) -> None:
    out.append("\n" + "=" * 70)
    out.append("LH-6b  within-class / within-rule-cell numeric correlation shrinkage")
    out.append("=" * 70)
    t_lab = train["health_condition"]
    e_lab = ext[TARGET_EXT]
    out.append(f"\n  {'scope':<28} {'mean|corr| train':>17} {'mean|corr| orig':>16}")
    a_t, _ = corr_offdiag(train)
    a_e, _ = corr_offdiag(ext)
    out.append(f"  {'marginal (all rows)':<28} {a_t:>17.4f} {a_e:>16.4f}")
    for cls in ["at-risk", "fit", "unhealthy"]:
        a_t, _ = corr_offdiag(train[t_lab == cls])
        a_e, _ = corr_offdiag(ext[e_lab == cls])
        out.append(f"  {'class=' + cls:<28} {a_t:>17.4f} {a_e:>16.4f}")
    # sharper: within one dense rule cell (complete drivers)
    for df, name, col in ((train, "train", "health_condition"), (ext, "orig", TARGET_EXT)):
        cell = (
            (sleep_bucket(df) == "b2")
            & (df["stress_level"] == "low")
            & (df["physical_activity_level"] == "active")
        )
        a, _ = corr_offdiag(df[cell])
        out.append(f"  {'rule-cell b2/low/active ' + name:<28} {a:>17.4f}")
    # strongest single pair in the original, tracked in train
    e_corr = ext[NUM_COLS].corr().abs()
    np.fill_diagonal(e_corr.to_numpy(), 0)
    i, j = np.unravel_index(np.nanargmax(e_corr.to_numpy()), e_corr.shape)
    c1, c2 = NUM_COLS[i], NUM_COLS[j]
    r_e = float(ext[[c1, c2]].corr().iloc[0, 1])
    r_t = float(train[[c1, c2]].corr().iloc[0, 1])
    out.append(f"\n  strongest original pair {c1} x {c2}: orig {r_e:+.4f} vs train {r_t:+.4f}")


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    ext = pd.read_csv(EXT)
    out: list[str] = [f"train {train.shape}, original {ext.shape}"]
    containment(train, ext, out)
    shrinkage(train, ext, out)
    report = "\n".join(out)
    print(report)
    (RESULTS_DIR / "probe_generator_fingerprint.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
