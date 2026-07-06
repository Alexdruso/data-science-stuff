"""Missingness-mechanism EDA: what does a missing value MEAN in this data?

The standing "missingness ≈ MCAR, unrecoverable" claim rests only on class-conditional null
rates. This runs the actual battery, each stage with an explicit null model:

  A. Rates & patterns — per-column rates train vs test; per-row missing-count vs the EXACT
     independence expectation (Poisson-binomial via convolution); top patterns obs/expected.
     Deviation ⇒ columns go missing together (pattern is a feature, not noise).
  B. Class dependence — per-column missing-rate by class + Cramér's V; balanced accuracy of
     predicting y from the 13 indicators alone (vs 1/3 chance).
  C. MAR test — per column j, small LGBM predicting missing_j from all OTHER columns
     (values + their indicators). MCAR ⇔ AUC ≈ 0.5 everywhere. THE definitive test.
  D. Indicator co-occurrence (phi correlations).
  E. MNAR test via the external source — the 50k source rows are complete, so if masking were
     value-dependent the OBSERVED train distribution would look truncated vs the source.
     Numerics: standardized quantile shifts (benchmark: all columns shift alike under the
     generator's global compression; an outlier on a high-missing column = MNAR signal).
     Categoricals: observed level shares vs source shares.

Standalone raw-CSV reads; no OOF/npy coupling.
Output → results/eda_missingness.txt + results/figures/miss_*.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET  # noqa: E402

COMP_DIR = Path(__file__).resolve().parents[1]
OUT = COMP_DIR / "results" / "eda_missingness.txt"
FIG_DIR = COMP_DIR / "results" / "figures"
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
COLS = NUM + CAT
MAR_SAMPLE = 200_000
QS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# validated reference palette (light mode)
C1, C2, C3 = "#2a78d6", "#1baf7a", "#eda100"  # categorical slots 1-3
RED, SURFACE, INK, INK2, GRID = "#e34948", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bwr_ref", [C1, "#f0efec", RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.axisbelow": True,
})

lines: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    say(f"  figure → {path}")


def poisson_binomial_pmf(rates: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0])
    for p in rates:
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def stage_a(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    say("\n=== A. rates & patterns ===")
    say(f"  {'column':24s} {'train':>7s} {'test':>7s}")
    r_tr = [tr[c].isna().mean() for c in COLS]
    r_te = [te[c].isna().mean() for c in COLS]
    for c, a, b in zip(COLS, r_tr, r_te):
        say(f"  {c:24s} {a:7.4f} {b:7.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(COLS))
    ax.barh(ypos + 0.19, np.array(r_tr) * 100, height=0.34, color=C1, label="train")
    ax.barh(ypos - 0.19, np.array(r_te) * 100, height=0.34, color=C2, label="test")
    ax.set_yticks(ypos, COLS)
    ax.invert_yaxis()
    ax.set_xlabel("missing rate (%)")
    ax.set_title("Missing rate per column — train vs test")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False)
    save_fig(fig, "miss_rates.png")

    miss = tr[COLS].isna()
    rates = miss.mean().to_numpy()
    pmf = poisson_binomial_pmf(rates)
    counts = miss.sum(axis=1).value_counts(normalize=True).sort_index()
    say("  per-row #missing: observed vs independence (Poisson-binomial):")
    ks = list(range(7))
    obs = [counts.get(k, 0.0) for k in ks]
    for k in ks:
        say(f"    k={k}: obs {obs[k]:8.5f}  expected {pmf[k]:8.5f}  "
            f"ratio {obs[k] / pmf[k] if pmf[k] > 0 else float('nan'):6.3f}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(ks))
    ax.bar(x - 0.19, obs, width=0.34, color=C1, label="observed")
    ax.bar(x + 0.19, pmf[: len(ks)], width=0.34, color=C3, label="independent-columns expectation")
    ax.set_yscale("log")
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("# missing values in row")
    ax.set_ylabel("share of rows (log)")
    ax.set_title("Rows by missing count — observed vs independence")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    save_fig(fig, "miss_row_counts.png")

    pat = miss.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    top = pat.value_counts().head(6)
    say(f"  distinct patterns: {pat.nunique()}  (13 cols → 8192 possible)")
    say("  top patterns (bitstring over " + ",".join(COLS) + "):")
    for p, n in top.items():
        exp = len(tr) * np.prod([rates[i] if b == "1" else 1 - rates[i] for i, b in enumerate(p)])
        say(f"    {p}  n={n:7d}  expected {exp:9.1f}  ratio {n / exp:6.3f}")


def stage_b(tr: pd.DataFrame) -> None:
    say("\n=== B. class dependence of missingness ===")
    y = tr[TARGET]
    classes = sorted(y.unique())
    say(f"  {'column':24s} " + " ".join(f"{c:>10s}" for c in classes) + "   CramersV")
    rel = np.zeros((len(COLS), len(classes)))
    for i, c in enumerate(COLS):
        m = tr[c].isna()
        overall = m.mean()
        by = [m[y == cl].mean() for cl in classes]
        rel[i] = [b / overall - 1 for b in by]
        ct = pd.crosstab(m, y).to_numpy()
        exp = ct.sum(1)[:, None] * ct.sum(0)[None, :] / ct.sum()
        chi2 = ((ct - exp) ** 2 / exp).sum()
        v = np.sqrt(chi2 / ct.sum())
        say(f"  {c:24s} " + " ".join(f"{r:10.4f}" for r in by) + f"   {v:8.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ypos = np.arange(len(COLS))
    for k, (cl, col) in enumerate(zip(classes, [C1, C2, C3])):
        ax.barh(ypos + (1 - k) * 0.26, rel[:, k] * 100, height=0.24, color=col, label=cl)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ypos, COLS)
    ax.invert_yaxis()
    ax.set_xlabel("class missing-rate vs column average (%)")
    ax.set_title("Does missingness depend on the label?")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "miss_class_dependence.png")

    X = tr[COLS].isna().astype(np.int8)
    yy = pd.factorize(y, sort=True)[0]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(tr), dtype=int)
    for tr_i, va_i in skf.split(X, yy):
        mdl = LGBMClassifier(n_estimators=60, num_leaves=31, class_weight="balanced",
                             random_state=42, verbose=-1)
        mdl.fit(X.iloc[tr_i], yy[tr_i])
        oof[va_i] = mdl.predict(X.iloc[va_i])
    say(f"  balanced acc, y ~ 13 indicators ONLY: {balanced_accuracy_score(yy, oof):.4f}  (chance 0.3333)")


def stage_c(tr: pd.DataFrame) -> None:
    say("\n=== C. MAR test — AUC of predicting missing_j from other columns (MCAR ⇒ ~0.5) ===")
    df = tr.sample(n=MAR_SAMPLE, random_state=42)
    for c in CAT:
        df[c] = df[c].astype("category")
    aucs: dict[str, float] = {}
    for j in COLS:
        target = df[j].isna().to_numpy().astype(int)
        if target.sum() < 500:
            say(f"  {j:24s} too few missing — skip")
            continue
        feats = [c for c in COLS if c != j]
        X = df[feats].copy()
        for c in feats:
            X[f"{c}__isna"] = df[c].isna().astype(np.int8)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        proba = np.zeros(len(df))
        for tr_i, va_i in skf.split(X, target):
            mdl = LGBMClassifier(n_estimators=120, num_leaves=63, random_state=42, verbose=-1)
            mdl.fit(X.iloc[tr_i], target[tr_i])
            proba[va_i] = mdl.predict_proba(X.iloc[va_i])[:, 1]
        aucs[j] = roc_auc_score(target, proba)
        say(f"  {j:24s} AUC = {aucs[j]:.4f}   (rate {target.mean():.4f})")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(aucs)
    vals = np.array([aucs[n] for n in names])
    ypos = np.arange(len(names))
    ax.barh(ypos, vals - 0.5, left=0.5, height=0.62,
            color=[RED if v >= 0.55 else C1 for v in vals])
    ax.axvline(0.5, color=INK2, lw=1)
    ax.axvline(0.55, color=RED, lw=0.8, ls="--")
    ax.annotate("MAR alert 0.55", (0.55, -0.7), color=RED, fontsize=8, ha="left")
    for yp, v in zip(ypos, vals):
        ax.annotate(f"{v:.3f}", (max(v, 0.502), yp), va="center", fontsize=8, color=INK2,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0.49, max(0.58, vals.max() + 0.02))
    ax.set_xlabel("AUC: predict missing_j from other columns")
    ax.set_title("MAR test — MCAR means every bar ≈ 0.5")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "miss_mar_auc.png")


def stage_d(tr: pd.DataFrame) -> None:
    say("\n=== D. indicator co-occurrence (phi) ===")
    m = tr[COLS].isna().astype(float)
    corr = m.corr().to_numpy()
    pairs = [(abs(corr[i, j]), COLS[i], COLS[j], corr[i, j])
             for i in range(len(COLS)) for j in range(i + 1, len(COLS))]
    for a, ci, cj, r in sorted(pairs, reverse=True)[:6]:
        say(f"  {ci:22s} × {cj:22s} phi = {r:+.4f}")

    vmax = max(0.02, np.abs(corr - np.eye(len(COLS))).max())
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr - np.eye(len(COLS)), cmap=DIVERGING,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(len(COLS)), COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(COLS)), COLS)
    ax.set_title("Missing-indicator co-occurrence (phi, diagonal removed)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="phi")
    save_fig(fig, "miss_cooccurrence.png")


def stage_e(tr: pd.DataFrame, ext: pd.DataFrame) -> None:
    say("\n=== E. MNAR test vs complete source — observed-train quantiles vs external ===")
    say("  (uniform shift across columns = generator compression; an OUTLIER on a")
    say("   high-missing column = value-dependent masking. shifts in EXTERNAL sd units)")
    say(f"  {'column':22s} {'miss%':>6s} " + " ".join(f"{f'q{int(q * 100)}':>7s}" for q in QS))
    shifts = np.zeros((len(NUM), len(QS)))
    for i, c in enumerate(NUM):
        sd = ext[c].std()
        shifts[i] = (tr[c].dropna().quantile(QS).to_numpy() - ext[c].quantile(QS).to_numpy()) / sd
        say(f"  {c:22s} {tr[c].isna().mean() * 100:5.1f}% "
            + " ".join(f"{v:+7.3f}" for v in shifts[i]))

    vmax = max(0.05, np.abs(shifts).max())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(shifts, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                   aspect="auto")
    ax.set_xticks(range(len(QS)), [f"q{int(q * 100)}" for q in QS])
    ax.set_yticks(range(len(NUM)), [f"{c}  ({tr[c].isna().mean() * 100:.0f}%)" for c in NUM])
    ax.set_title("Observed train vs complete source — quantile shift (sd units)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="shift (ext sd)")
    save_fig(fig, "miss_mnar_quantiles.png")

    say("  categorical level shares, observed train vs external:")
    for c in CAT:
        a = tr[c].dropna().value_counts(normalize=True)
        b = ext[c].value_counts(normalize=True)
        parts = [f"{lv}: {a.get(lv, 0):.3f}/{b.get(lv, 0):.3f}" for lv in sorted(b.index)]
        say(f"  {c:24s} " + "  ".join(parts))

    say("  P(stress_level missing | other drivers observed):")
    sub = tr.dropna(subset=["sleep_duration", "physical_activity_level"])
    m = sub["stress_level"].isna()
    for act in ["sedentary", "moderate", "active"]:
        say(f"    activity={act:10s} → {m[sub['physical_activity_level'] == act].mean():.4f}")
    for lo, hi in [(0, 6), (6, 7), (7, 12)]:
        sel = (sub["sleep_duration"] >= lo) & (sub["sleep_duration"] < hi)
        say(f"    sleep∈[{lo},{hi})      → {m[sel].mean():.4f}")


RULES = [("water_intake", "gender"), ("diet_type", "gender"),
         ("bmi", "sleep_quality"), ("physical_activity_level", "smoking_alcohol")]


def stage_f(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    say("\n=== F. THE FINDING — mask mechanism differs between train and test ===")
    say("  In TRAIN, four masks are single-column trigger rules (hard zeros off-trigger).")
    say("  In TEST, the same masks are UNIFORM at the same marginal rate:")
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
    for ax, (mask_col, trig) in zip(axes, RULES):
        r_tr = tr.groupby(trig, observed=True)[mask_col].apply(lambda s: s.isna().mean())
        r_te = te.groupby(trig, observed=True)[mask_col].apply(lambda s: s.isna().mean())
        say(f"  missing({mask_col}) by {trig}:")
        for lv in r_tr.index:
            say(f"     {lv:12s} train {r_tr[lv]:.4f}   test {r_te[lv]:.4f}")
        x = np.arange(len(r_tr))
        ax.bar(x - 0.19, r_tr.to_numpy() * 100, width=0.34, color=C1, label="train")
        ax.bar(x + 0.19, r_te.to_numpy() * 100, width=0.34, color=C2, label="test")
        ax.set_xticks(x, r_tr.index, rotation=20)
        ax.set_title(f"missing({mask_col})\nby {trig}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("missing rate (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Train masks are trigger-conditional; test masks are uniform — a mechanism shift",
                 fontsize=10, fontweight="bold", y=1.14)
    save_fig(fig, "miss_mechanism_shift.png")

    cols = COLS
    pt = set(map(tuple, tr[cols].isna().to_numpy()))
    pe = pd.Series(map(tuple, te[cols].isna().to_numpy()))
    unseen = ~pe.isin(pt)
    say(f"  test rows whose missingness PATTERN never occurs in train: "
        f"{unseen.sum()} ({unseen.mean() * 100:.3f}%)")
    say("  consequences: (1) trees bind train-only NaN↔trigger couplings (e.g. water-NaN ⇒ female)")
    say("  that misfire on ~2/3 of test NaN rows incl. the activity driver; (2) ~1.3% of test rows")
    say("  are out-of-distribution patterns; (3) explains part of adv-AUC 0.65 and the water-drop")
    say("  flip fragility. Repair candidates: impute-the-4 / uniform re-masking. See CLAUDE.md.")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    tr = pd.read_csv(COMP_DIR / "data" / "train.csv")
    te = pd.read_csv(COMP_DIR / "data" / "test.csv")
    ext = pd.read_csv(COMP_DIR / "data" / "external" / "student_health_dataset_50k.csv")
    say(f"train {tr.shape}, test {te.shape}, external {ext.shape}")
    stage_a(tr, te)
    stage_b(tr)
    stage_c(tr)
    stage_d(tr)
    stage_e(tr, ext)
    stage_f(tr, te)
    OUT.write_text("\n".join(lines) + "\n")
    say(f"\nreport → {OUT}")


if __name__ == "__main__":
    main()
