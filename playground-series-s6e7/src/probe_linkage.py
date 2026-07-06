"""Clone-linkage probe: are competition rows noisy copies of the 50k source rows?

If the synthetic generator cloned source rows (exactly or +noise), matching a competition row
back to its source row recovers that row's MISSING drivers — the only mechanism left that could
beat the 0.886 missing-region ceiling. Three stages, each with an explicit null:

  A. EXACT match on all 7 numerics (joint cardinality ~1e19 → chance collisions ≈ 0 across
     690k×50k pairs; any nontrivial count = linkage).
  B. NOISY-clone structure: NN distance from competition rows to external in standardized
     7-numeric space, vs two nulls — external→external leave-one-out spacing (density-sample
     expectation) and a column-permuted competition sample (no-structure floor). Clones show up
     as a mass of distances far BELOW the LOO spacing.
  C. USABLE recovery: mask stress_level on complete train rows, NN-match on the other numerics
     (+ exact remaining categoricals), read stress_level/label off the match. Control = the SAME
     procedure matching into train itself (leave-self-out): external linkage is only real if the
     external match beats the train self-match (otherwise it's just density, not identity).

Standalone: reads raw CSVs, never touches OOF/test npy arrays (no row-order coupling).
Output → results/probe_linkage.txt. Runtime ~2-3 min on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET  # noqa: E402

COMP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = COMP_DIR / "data"
OUT = COMP_DIR / "results" / "probe_linkage.txt"

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
RNG = np.random.default_rng(42)
SAMPLE = 25_000

lines: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def exact_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # fixed-format strings so float repr noise can't break equality
    parts = [df[c].map(lambda v: f"{v:.4f}") for c in cols]
    return parts[0].str.cat(parts[1:], sep="|")


def stage_a(tr: pd.DataFrame, te: pd.DataFrame, ext: pd.DataFrame) -> None:
    say("\n=== Stage A — EXACT numeric match (7 cols, joint cardinality ~1e19) ===")
    ext_keys = set(exact_key(ext, NUM))
    for name, df in [("train", tr), ("test", te)]:
        comp = df.dropna(subset=NUM)
        hits = exact_key(comp, NUM).isin(ext_keys).sum()
        say(f"  {name}: {len(comp):7d} complete-numeric rows → exact matches to external: {hits}")
    # partial-exactness: does ANY single column survive un-noised? (clone+noise on a subset)
    say("  per-column exact-value overlap (share of complete train rows whose value exists in ext):")
    comp = tr.dropna(subset=NUM).sample(n=SAMPLE, random_state=42)
    for c in NUM:
        share = comp[c].isin(set(ext[c])).mean()
        say(f"    {c:22s} {share:6.1%}   (high = shared quantization grid, NOT linkage by itself)")
    pair_cols = ["step_count", "bmi"]  # ~2e7 joint values: pair-exactness is already telling
    ext_pair = set(exact_key(ext, pair_cols))
    hits = exact_key(comp, pair_cols).isin(ext_pair).sum()
    say(f"  (step_count,bmi) pair-exact matches in {SAMPLE} train rows: {hits} "
        f"(chance ≈ {SAMPLE * len(ext) / (13601 * 1627):,.0f} if values were uniform)")


def zspace(df: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((df[NUM].to_numpy(dtype=np.float64) - mu) / sd).astype(np.float32)


def stage_b(tr: pd.DataFrame, te: pd.DataFrame, ext: pd.DataFrame) -> None:
    say("\n=== Stage B — noisy-clone structure (NN distance, standardized 7-numeric space) ===")
    mu = ext[NUM].to_numpy(dtype=np.float64).mean(axis=0)
    sd = ext[NUM].to_numpy(dtype=np.float64).std(axis=0)
    X_ext = zspace(ext, mu, sd)
    nn = NearestNeighbors(n_neighbors=2).fit(X_ext)
    loo = nn.kneighbors(X_ext)[0][:, 1]  # leave-one-out self-spacing of the source cloud

    def report(tag: str, X: np.ndarray) -> None:
        d = nn.kneighbors(X, n_neighbors=1)[0][:, 0]
        q = np.percentile(d, [1, 5, 25, 50])
        say(f"  {tag:18s} NN-dist p1={q[0]:.4f} p5={q[1]:.4f} p25={q[2]:.4f} p50={q[3]:.4f}"
            f"   frac < ext-LOO-p5({np.percentile(loo, 5):.4f}): {(d < np.percentile(loo, 5)).mean():.4f}")

    say(f"  ext→ext LOO       p1={np.percentile(loo, 1):.4f} p5={np.percentile(loo, 5):.4f} "
        f"p25={np.percentile(loo, 25):.4f} p50={np.percentile(loo, 50):.4f}   ← density expectation")
    for tag, df in [("train→ext", tr), ("test→ext", te)]:
        comp = df.dropna(subset=NUM).sample(n=SAMPLE, random_state=42)
        report(tag, zspace(comp, mu, sd))
    # no-structure floor: independently permute each column of the train sample
    comp = tr.dropna(subset=NUM).sample(n=SAMPLE, random_state=42)
    # independent permutation PER COLUMN (distinct seeds) — same-seed shuffles would merely
    # reorder whole rows and reproduce the real sample's distances exactly
    perm = pd.DataFrame({c: comp[c].sample(frac=1, random_state=100 + i).to_numpy()
                         for i, c in enumerate(NUM)})
    report("permuted-null→ext", zspace(perm, mu, sd))
    say("  verdict rule: clones ⇒ train/test curves sit FAR BELOW ext-LOO (frac≫0.05); "
        "density-sampled synthetic ⇒ ≈ LOO; permuted-null shows the no-structure ceiling.")


def stage_c(tr: pd.DataFrame, ext: pd.DataFrame) -> None:
    say("\n=== Stage C — usable recovery: mask stress_level, read it off the matched row ===")
    feats = [c for c in NUM if c != "water_intake"]  # water is pure noise; drop from the metric
    comp = tr.dropna(subset=NUM + CAT).sample(n=SAMPLE, random_state=42)
    mu = ext[feats].to_numpy(dtype=np.float64).mean(axis=0)
    sd = ext[feats].to_numpy(dtype=np.float64).std(axis=0)

    def z(df: pd.DataFrame) -> np.ndarray:
        return ((df[feats].to_numpy(dtype=np.float64) - mu) / sd).astype(np.float32)

    # match into EXTERNAL
    nn_ext = NearestNeighbors(n_neighbors=1).fit(z(ext))
    idx_ext = nn_ext.kneighbors(z(comp), n_neighbors=1)[1][:, 0]
    acc_ext_stress = (ext["stress_level"].to_numpy()[idx_ext] == comp["stress_level"].to_numpy()).mean()
    acc_ext_label = (ext[TARGET].to_numpy()[idx_ext] == comp[TARGET].to_numpy()).mean()

    # control: SAME procedure into train itself (2nd neighbor = leave-self-out)
    pool = tr.dropna(subset=NUM + CAT)
    nn_tr = NearestNeighbors(n_neighbors=2).fit(z(pool))
    idx_tr = nn_tr.kneighbors(z(comp), n_neighbors=2)[1][:, 1]
    acc_tr_stress = (pool["stress_level"].to_numpy()[idx_tr] == comp["stress_level"].to_numpy()).mean()
    acc_tr_label = (pool[TARGET].to_numpy()[idx_tr] == comp[TARGET].to_numpy()).mean()

    marg_stress = comp["stress_level"].value_counts(normalize=True).max()
    marg_label = comp[TARGET].value_counts(normalize=True).max()
    say(f"  stress_level recovery: ext-match {acc_ext_stress:.4f} | train-self-match {acc_tr_stress:.4f}"
        f" | majority {marg_stress:.4f}")
    say(f"  label       recovery: ext-match {acc_ext_label:.4f} | train-self-match {acc_tr_label:.4f}"
        f" | majority {marg_label:.4f}")
    say("  verdict rule: linkage is USABLE only if ext-match ≫ train-self-match "
        "(same-density matching must not explain it).")


def main() -> None:
    ext = pd.read_csv(DATA_DIR / "external" / "student_health_dataset_50k.csv")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    say(f"external {ext.shape}, train {tr.shape}, test {te.shape}")
    stage_a(tr, te, ext)
    stage_b(tr, te, ext)
    stage_c(tr, ext)
    OUT.write_text("\n".join(lines) + "\n")
    say(f"\nreport → {OUT}")


if __name__ == "__main__":
    main()
