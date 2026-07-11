"""Day-9 leak-hunter probes (notes/ideas_day9_personas.md, LEAK HUNTER 2/3/5).

All on the RAW surface (no repair) — these probe the actual generator, not the
training emulation. Report -> results/probe_leaks_day9.txt.

LH-2  Test-head segment check: test masks are blockwise in id (eda_id G). Are
      any test segments still in TRAIN trigger mode (mask conditional on the
      trigger column) rather than uniform? Per 2k-id segment x 4 mechanism
      columns: contingency of mask x trigger level + binomial p-value of the
      observed non-trigger miss count under the segment's pooled rate. A
      train-mode segment (trigger-only masking) => segment-routed submission.

LH-3  Gender step at the train/test boundary (id 690,088): gender shares in
      5k-id bins across the last/first 100k; a STEP at the boundary (>> bin sd)
      = two generator runs; smooth drift = within-run drift.

LH-5  Label-noise burstiness: on rule-DETERMINED train rows (rule_features
      deduction, drivers observed), disagreement label != deduced is generator
      noise. Per 5k-id bin: rate vs binomial null + CUSUM change-point scan.
      Bursty noise => drop/downweight noisy id segments at training time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features
from rule_features import rule_label_set
from train_common import DATA_DIR, RESULTS_DIR

# train-side deterministic trigger rules (eda_missingness F)
TRIGGERS: dict[str, tuple[str, str]] = {
    "water_intake": ("gender", "female"),
    "diet_type": ("gender", "other"),
    "bmi": ("sleep_quality", "good"),
    "physical_activity_level": ("smoking_alcohol", "occasional"),
}
SEG = 2_000  # LH-2 segment width in test rows (id-ordered)
BIN = 5_000  # LH-3/LH-5 bin width


def lh2_segments(test: pd.DataFrame, out: list[str]) -> None:
    out.append("=" * 70)
    out.append(f"LH-2  test segments in TRAIN trigger mode? (segment={SEG} rows)")
    out.append("=" * 70)
    n_seg = int(np.ceil(len(test) / SEG))
    seg_id = np.arange(len(test)) // SEG
    any_hit = False
    for col, (trig_col, trig_val) in TRIGGERS.items():
        obs = test[trig_col].notna()
        miss = test[col].isna()
        is_trig = (test[trig_col] == trig_val) & obs
        df = pd.DataFrame(
            {"seg": seg_id[obs], "miss": miss[obs], "trig": is_trig[obs]}
        )
        g = df.groupby("seg").agg(
            n=("miss", "size"),
            n_trig=("trig", "sum"),
            miss_trig=("miss", lambda s: 0),  # placeholder, filled below
        )
        # counts per segment split by trigger membership
        ct = df.groupby(["seg", "trig"])["miss"].agg(["sum", "size"]).unstack("trig")
        miss_nt = ct[("sum", False)].fillna(0).astype(int)
        n_nt = ct[("size", False)].fillna(0).astype(int)
        miss_t = ct[("sum", True)].fillna(0).astype(int)
        n_t = ct[("size", True)].fillna(0).astype(int)
        pooled = (miss_nt + miss_t) / (n_nt + n_t).clip(lower=1)
        # train-mode signature: zero non-trigger misses where >=5 expected,
        # while the trigger group carries the mass
        expect_nt = pooled * n_nt
        p_zero = stats.binom.cdf(miss_nt, n_nt, pooled.clip(lower=1e-9))
        cand = (miss_nt == 0) & (expect_nt >= 5) & (miss_t > 0)
        rate_t = (miss_t / n_t.clip(lower=1)).mean()
        rate_nt = (miss_nt / n_nt.clip(lower=1)).mean()
        out.append(
            f"\n{col:<26} trigger={trig_col}={trig_val}"
            f"  mean rate trig={rate_t:.4f} nontrig={rate_nt:.4f}"
        )
        out.append(
            f"  segments={n_seg}  train-mode candidates "
            f"(0 nontrig misses, >=5 expected): {int(cand.sum())}"
        )
        if cand.sum() > 0:
            any_hit = True
            for s in cand[cand].index[:10]:
                lo, hi = int(test['id'].iloc[s * SEG]), int(
                    test['id'].iloc[min((s + 1) * SEG, len(test)) - 1]
                )
                out.append(
                    f"    seg {s} ids [{lo}, {hi}]: trig {int(miss_t[s])}/{int(n_t[s])}"
                    f" nontrig 0/{int(n_nt[s])} (P0={p_zero[s]:.2e})"
                )
        # also: min binomial-tail across segments (multiple-testing aware)
        with np.errstate(divide="ignore"):
            worst = float(p_zero[expect_nt >= 5].min()) if (expect_nt >= 5).any() else 1.0
        out.append(f"  min P(<=obs nontrig | pooled) over testable segs: {worst:.2e}"
                   f"  (Bonferroni x{int((expect_nt >= 5).sum())})")
    out.append(
        "\nVERDICT LH-2: " + ("*** TRAIN-MODE SEGMENTS FOUND — route them ***"
                              if any_hit else
                              "no test segment shows trigger-mode masking (all uniform).")
    )


def lh3_gender_step(train: pd.DataFrame, test: pd.DataFrame, out: list[str]) -> None:
    out.append("\n" + "=" * 70)
    out.append(f"LH-3  gender step at the train/test boundary (bin={BIN} ids)")
    out.append("=" * 70)
    tail = train.tail(100_000)
    head = test.head(100_000)
    for level in ["female", "male", "other"]:
        rates = []
        for name, df in [("train", tail), ("test", head)]:
            obs = df[df["gender"].notna()]
            b = (obs["id"].to_numpy() // BIN).astype(int)
            share = pd.Series((obs["gender"] == level).to_numpy()).groupby(b).mean()
            rates.append((name, share))
        tr_s, te_s = rates[0][1], rates[1][1]
        step = float(te_s.iloc[0] - tr_s.iloc[-1])
        sd_within = float(np.sqrt((tr_s.diff().dropna().var() + te_s.diff().dropna().var()) / 2))
        out.append(
            f"  {level:<8} last-train-bin {tr_s.iloc[-1]:.4f}  first-test-bin {te_s.iloc[0]:.4f}"
            f"  step {step:+.4f}  vs bin-to-bin diff sd {sd_within:.4f}"
            f"  |z|={abs(step) / max(sd_within, 1e-9):.1f}"
        )
        out.append(
            f"          train-tail trend {tr_s.iloc[0]:.4f}->{tr_s.iloc[-1]:.4f}, "
            f"test-head trend {te_s.iloc[0]:.4f}->{te_s.iloc[-1]:.4f}"
        )


def lh5_burstiness(train: pd.DataFrame, out: list[str]) -> None:
    out.append("\n" + "=" * 70)
    out.append(f"LH-5  label-noise burstiness on rule-DETERMINED rows (bin={BIN} ids)")
    out.append("=" * 70)
    rs = rule_label_set(train)
    det = ~rs.str.contains(r"\|")
    sub = train[det].copy()
    sub["deduced"] = rs[det]
    dis = (sub[TARGET] != sub["deduced"]).to_numpy()
    out.append(
        f"  determined rows: {det.sum():,} ({det.mean():.1%})  "
        f"overall disagreement (= generator noise) {dis.mean():.4%}"
    )
    b = (sub["id"].to_numpy() // BIN).astype(int)
    g = pd.DataFrame({"b": b, "d": dis}).groupby("b")["d"].agg(["mean", "size"])
    p0 = dis.mean()
    exp_sd = float(np.sqrt(p0 * (1 - p0) / g["size"].mean()))
    obs_sd = float(g["mean"].std())
    out.append(
        f"  bins={len(g)}  rate sd across bins {obs_sd:.5f} vs binomial-null sd {exp_sd:.5f}"
        f"  (ratio {obs_sd / exp_sd:.2f}; ~1.0 = homogeneous noise)"
    )
    # CUSUM change-point scan with a permutation null
    x = (g["mean"] - p0).to_numpy() * g["size"].to_numpy()
    cusum = np.abs(np.cumsum(x))
    stat = float(cusum.max())
    rng = np.random.default_rng(0)
    null = np.array([
        np.abs(np.cumsum(rng.permutation(x))).max() for _ in range(2000)
    ])
    pval = float((null >= stat).mean())
    kbin = int(cusum.argmax())
    out.append(
        f"  CUSUM max at bin {kbin} (id ~{g.index[kbin] * BIN:,}): stat {stat:.1f}, "
        f"permutation p = {pval:.3f}"
    )
    top = g["mean"].sort_values(ascending=False).head(5)
    out.append("  top-5 noisiest bins: " + ", ".join(
        f"id~{i * BIN // 1000}k:{v:.3%}" for i, v in top.items()
    ))
    out.append(
        "\nVERDICT LH-5: " + ("*** BURSTY — segment-weighted training is live ***"
                              if pval < 0.01 and obs_sd / exp_sd > 1.5 else
                              "noise is homogeneous in id — no segment lever.")
    )


def main() -> None:
    train = build_features(pl.read_csv(DATA_DIR / "train.csv")).to_pandas()
    test = build_features(pl.read_csv(DATA_DIR / "test.csv")).to_pandas()
    out: list[str] = []
    lh2_segments(test, out)
    lh3_gender_step(train, test, out)
    lh5_burstiness(train, out)
    report = "\n".join(out)
    print(report)
    (RESULTS_DIR / "probe_leaks_day9.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
