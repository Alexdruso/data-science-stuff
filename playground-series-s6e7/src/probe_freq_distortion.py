"""Class-conditional frequency-distortion pre-check (Day-10, GF #4).

CTGAN-class generators sample with a conditional vector: per-value frequency
distortion (synth vs original) then depends on the CLASS conditioned during
sampling. If distortion is class-conditional, per-column frequency features
(log-count, freq-ratio-to-original, row distortion sum) carry label signal
that TE posteriors don't; if distortion is class-independent, the feature
family is dead and we save the build.

Test, per column: for each value v (count >= MIN_N in both datasets), a 2x3
chi-square of dataset-membership x class given v — under class-independent
distortion n_synth(v,c) is proportional to n_orig(v,c). BH-FDR across values
within a column. water_intake is the built-in control: it carries ~zero label
signal, so any class-conditional distortion there is a pure generator
fingerprint, not label noise leaking through P(y|v).

Run: python src/probe_freq_distortion.py -> results/probe_freq_distortion.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES, TARGET
from train_common import DATA_DIR, RESULTS_DIR, load_dataset

EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"
MIN_N = 200


def crosstab(values: pd.Series, y: np.ndarray, n_classes: int) -> pd.DataFrame:
    df = pd.DataFrame({"v": values.to_numpy(), "y": y})
    ct = df.groupby(["v", "y"]).size().unstack(fill_value=0)
    return ct.reindex(columns=range(n_classes), fill_value=0)


def main() -> None:
    ds = load_dataset()
    ext = pd.read_csv(EXT_CSV)
    y_ext = ext[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    lines = [
        f"class-conditional frequency distortion, values with n>={MIN_N} both sides",
        f"synth train {len(ds.y)} rows vs original {len(ext)} rows",
    ]
    summary = []
    for col in ds.feature_cols:
        ct_s = crosstab(ds.train[col].astype(str), ds.y, len(CLASSES))
        ct_o = crosstab(ext[col].astype(str), y_ext, len(CLASSES))
        common = ct_s.index.intersection(ct_o.index)
        common = [
            v
            for v in common
            if ct_s.loc[v].sum() >= MIN_N and ct_o.loc[v].sum() >= MIN_N
        ]
        if not common:
            lines.append(f"{col:<28} no values clear n>={MIN_N} on both sides")
            continue
        pvals, effects = [], []
        for v in common:
            table = np.stack([ct_o.loc[v].to_numpy(), ct_s.loc[v].to_numpy()])
            table = table[:, table.sum(axis=0) > 0]
            chi2, p, _, _ = chi2_contingency(table)
            pvals.append(p)
            # effect: TV distance between the two class distributions at v
            po = table[0] / table[0].sum()
            ps = table[1] / table[1].sum()
            effects.append(0.5 * np.abs(po - ps).sum())
        q = false_discovery_control(np.array(pvals))
        sig = q < 0.05
        order = np.argsort(np.array(pvals))
        top = ", ".join(
            f"{common[i]}(TV {effects[i]:.3f}, q {q[i]:.1e})" for i in order[:3]
        )
        lines.append(
            f"{col:<28} {int(sig.sum()):>3}/{len(common):<3} values FDR-sig   "
            f"median TV {np.median(effects):.4f}   top: {top}"
        )
        summary.append((col, sig.mean(), float(np.median(effects))))

    frac = np.mean([s[1] for s in summary]) if summary else 0.0
    lines.append(
        f"\nverdict: mean FDR-sig fraction {frac:.2%} — "
        + (
            "class-conditional distortion PRESENT -> freq-distortion features live"
            if frac > 0.10
            else "distortion ~class-independent -> freq-distortion features dead"
        )
    )
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "probe_freq_distortion.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
