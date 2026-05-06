"""Distribution shift analysis for PS S6E5.

Sections:
  1. Categorical set membership (Race/Year/Driver overlap)
  2. Adversarial validation (LGBM train-vs-test classifier)
  3. Per-year per-model OOF breakdown
  4. Expected LB estimate from test year distribution
  5. Test prediction distribution vs OOF (per model)
  6. Numeric marginal KS tests
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "distribution_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from features import TARGET, build_features

NUM_FEATURES = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]
CAT_FEATURES = ["Driver", "Compound", "Race", "Year"]
MODELS = ["lgbm", "xgboost", "catboost", "mlp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sep(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


def _load_oof_arrays(
    train_sorted: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    y = train_sorted[TARGET].to_numpy()
    year = train_sorted["Year"].to_numpy()
    oofs: dict[str, np.ndarray] = {}
    for m in MODELS:
        path = RESULTS_DIR / f"oof_{m}.npy"
        if path.exists():
            oofs[m] = np.load(path)
    ens_path = RESULTS_DIR / "oof_ensemble.npy"
    if ens_path.exists():
        oofs["ensemble"] = np.load(ens_path)
    return y, year, oofs


def _load_test_arrays() -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for m in MODELS:
        path = RESULTS_DIR / f"test_{m}.npy"
        if path.exists():
            preds[m] = np.load(path)
    return preds


# ---------------------------------------------------------------------------
# Section 1: Categorical set membership
# ---------------------------------------------------------------------------


def section1_set_membership(train: pl.DataFrame, test: pl.DataFrame) -> dict[int, dict]:
    _sep("Section 1 — Categorical Set Membership")

    # Year distribution
    def year_dist(df: pl.DataFrame, name: str) -> pl.DataFrame:
        vc = (
            df.group_by("Year")
            .agg(pl.len().alias("N"))
            .with_columns((pl.col("N") / pl.col("N").sum() * 100).alias("pct"))
            .sort("Year")
        )
        print(f"\n{name} Year distribution:")
        for row in vc.iter_rows(named=True):
            print(f"  {row['Year']}  N={row['N']:>7,}  {row['pct']:5.1f}%")
        return vc

    train_year = year_dist(train, "TRAIN")
    test_year = year_dist(test, "TEST")

    # Build test year fractions (for section 4)
    test_year_frac: dict[int, float] = {
        row["Year"]: row["pct"] / 100.0
        for row in test_year.iter_rows(named=True)
    }

    # Compute overlap for different tuple granularities
    configs = [
        ("Year", ["Year"]),
        ("Race", ["Race"]),
        ("(Race, Year)", ["Race", "Year"]),
        ("Driver", ["Driver"]),
        ("(Driver, Race, Year)", ["Driver", "Race", "Year"]),
    ]

    print("\nSet membership summary (unique tuples):")
    print(f"  {'Granularity':<25} {'Train-only':>12} {'Test-only':>12} {'Shared':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    results: dict[str, dict] = {}
    for label, cols in configs:
        train_set = set(map(tuple, train.select(cols).unique().to_numpy().tolist()))
        test_set = set(map(tuple, test.select(cols).unique().to_numpy().tolist()))
        train_only = train_set - test_set
        test_only = test_set - train_set
        shared = train_set & test_set
        print(
            f"  {label:<25} {len(train_only):>12,} {len(test_only):>12,} {len(shared):>10,}"
        )
        results[label] = {
            "train_only": train_only,
            "test_only": test_only,
            "shared": shared,
        }

    # Flag any test-only race-years (unseen race contexts)
    ry = results["(Race, Year)"]
    if ry["test_only"]:
        print(f"\n  ⚠ {len(ry['test_only'])} (Race, Year) tuples in TEST but not TRAIN:")
        for t in sorted(ry["test_only"])[:20]:
            print(f"      {t}")
    else:
        print("\n  ✓ No (Race, Year) tuples exclusively in test — all contexts seen in training.")

    drv = results["Driver"]
    if drv["test_only"]:
        print(f"\n  ⚠ {len(drv['test_only'])} Drivers in TEST but not TRAIN:")
        for d in sorted(drv["test_only"])[:10]:
            print(f"      {d[0]}")
    else:
        print("  ✓ No Drivers exclusively in test.")

    return test_year_frac  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Section 2: Adversarial validation
# ---------------------------------------------------------------------------


def section2_adversarial(train: pl.DataFrame, test: pl.DataFrame) -> None:
    _sep("Section 2 — Adversarial Validation")

    # Use raw numeric + simple categorical encoding; drop target
    shared_cols = [c for c in train.columns if c in test.columns and c != "id"]

    train_pd = train.select(shared_cols).to_pandas()
    test_pd = test.select(shared_cols).to_pandas()

    for col in CAT_FEATURES:
        if col in train_pd.columns:
            combined_cats = pd.Categorical(
                pd.concat([train_pd[col], test_pd[col]], ignore_index=True)
            )
            train_pd[col] = pd.Categorical(train_pd[col], categories=combined_cats.categories).codes
            test_pd[col] = pd.Categorical(test_pd[col], categories=combined_cats.categories).codes

    X = pd.concat([train_pd, test_pd], ignore_index=True)
    y_adv = np.array([0] * len(train_pd) + [1] * len(test_pd))

    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_adv = np.zeros(len(X))
    importances = np.zeros(X.shape[1])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_adv), 1):
        clf.fit(X.iloc[tr_idx], y_adv[tr_idx])
        oof_adv[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        importances += clf.feature_importances_ / 5

    adv_auc = roc_auc_score(y_adv, oof_adv)
    print(f"\nAdversarial AUC: {adv_auc:.4f}")
    if adv_auc < 0.52:
        print("  → Distributions are essentially identical (AUC ≈ 0.5).")
    elif adv_auc < 0.60:
        print("  → Mild distribution shift.")
    else:
        print("  → Strong distribution shift — test is distinguishable from train.")

    # Feature importance
    feat_order = np.argsort(importances)[::-1]
    print("\nTop 15 features driving train/test separation:")
    print(f"  {'Feature':<35} {'Importance':>12}")
    print(f"  {'-'*35} {'-'*12}")
    for i in feat_order[:15]:
        print(f"  {X.columns[i]:<35} {importances[i]:>12.1f}")

    # Plot
    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 5))
    feats = [X.columns[i] for i in feat_order[:top_n]][::-1]
    imps = importances[feat_order[:top_n]][::-1]
    ax.barh(feats, imps, color="steelblue")
    ax.set_xlabel("Mean feature importance")
    ax.set_title(f"Adversarial Validation — Feature Importance\n(AUC={adv_auc:.4f})")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "adversarial_importance.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved → {OUT_DIR / 'adversarial_importance.png'}")


# ---------------------------------------------------------------------------
# Section 3: Per-year per-model OOF breakdown
# ---------------------------------------------------------------------------


def section3_per_year_oof(
    train_sorted: pl.DataFrame,
    y: np.ndarray,
    year: np.ndarray,
    oofs: dict[str, np.ndarray],
) -> dict[str, dict[int, float]]:
    _sep("Section 3 — Per-Year Per-Model OOF Breakdown")

    years = sorted(np.unique(year).tolist())
    model_names = [m for m in MODELS if m in oofs] + (
        ["ensemble"] if "ensemble" in oofs else []
    )

    # Header
    col_w = 10
    header = f"  {'Year':>6}  {'N':>8}  {'pit%':>6}"
    for m in model_names:
        header += f"  {m.upper():>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_year_aucs: dict[str, dict[int, float]] = {m: {} for m in model_names}
    year_ns: dict[int, int] = {}

    for yr in years:
        mask = year == yr
        n = int(mask.sum())
        year_ns[yr] = n
        pit_rate = float(y[mask].mean() * 100)
        row = f"  {yr:>6}  {n:>8,}  {pit_rate:>5.1f}%"
        for m in model_names:
            if m in oofs and oofs[m].shape[0] == len(y):
                auc = roc_auc_score(y[mask], oofs[m][mask])
            else:
                auc = float("nan")
            per_year_aucs[m][yr] = auc
            row += f"  {auc:>{col_w}.4f}"
        print(row)

    # Overall row
    row = f"  {'OVERALL':>6}  {len(y):>8,}  {float(y.mean()*100):>5.1f}%"
    for m in model_names:
        if m in oofs:
            row += f"  {roc_auc_score(y, oofs[m]):>{col_w}.4f}"
    print(row)

    # OOF inflation: overall minus non-2023 weighted average
    print("\n  OOF Inflation (overall OOF minus non-2023 weighted AUC):")
    non_2023_years = [yr for yr in years if yr != 2023]
    total_non_2023 = sum(year_ns[yr] for yr in non_2023_years)
    for m in model_names:
        if m not in oofs:
            continue
        overall = roc_auc_score(y, oofs[m])
        weighted = sum(
            (year_ns[yr] / total_non_2023) * per_year_aucs[m][yr]
            for yr in non_2023_years
        )
        inflation = overall - weighted
        print(f"    {m.upper():<12}  overall={overall:.4f}  non-2023 wtd={weighted:.4f}  inflation={inflation:+.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    width = 0.8 / len(model_names)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, m in enumerate(model_names):
        vals = [per_year_aucs[m].get(yr, float("nan")) for yr in years]
        ax.bar(x + i * width, vals, width, label=m.upper(), color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([str(yr) for yr in years])
    ax.set_ylim(0.88, 0.97)
    ax.set_ylabel("OOF AUC")
    ax.set_title("Per-Year OOF AUC by Model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_year_auc.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved → {OUT_DIR / 'per_year_auc.png'}")

    return per_year_aucs


# ---------------------------------------------------------------------------
# Section 4: Expected LB estimate
# ---------------------------------------------------------------------------


def section4_expected_lb(
    y: np.ndarray,
    year: np.ndarray,
    oofs: dict[str, np.ndarray],
    per_year_aucs: dict[str, dict[int, float]],
    test_year_frac: dict[int, float],
) -> None:
    _sep("Section 4 — Expected LB Estimate")

    print(f"\n  Test year fractions used for weighting:")
    for yr, frac in sorted(test_year_frac.items()):
        print(f"    {yr}: {frac*100:.1f}%")

    model_names = [m for m in MODELS if m in oofs] + (
        ["ensemble"] if "ensemble" in oofs else []
    )

    print(f"\n  {'Model':<12}  {'OOF AUC':>10}  {'Expected LB':>12}  {'Pred. Drop':>12}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}")

    for m in model_names:
        if m not in oofs:
            continue
        overall_oof = roc_auc_score(y, oofs[m])
        expected_lb = sum(
            test_year_frac.get(yr, 0.0) * per_year_aucs[m].get(yr, overall_oof)
            for yr in test_year_frac
        )
        drop = expected_lb - overall_oof
        print(
            f"  {m.upper():<12}  {overall_oof:>10.4f}  {expected_lb:>12.4f}  {drop:>+12.4f}"
        )

    print(
        "\n  Note: Expected LB reweights per-year OOF AUCs by test year distribution."
        "\n  A negative predicted drop = OOF inflated relative to expected test performance."
    )


# ---------------------------------------------------------------------------
# Section 5: Test prediction distribution vs OOF
# ---------------------------------------------------------------------------


def section5_pred_distributions(
    y: np.ndarray,
    oofs: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
) -> None:
    _sep("Section 5 — Test Prediction Distribution vs OOF")

    model_names = [m for m in MODELS if m in oofs and m in test_preds]
    if not model_names:
        print("  No matching OOF + test arrays found.")
        return

    print(f"\n  {'Model':<12}  {'Split':<6}  {'Mean':>8}  {'Std':>8}  {'P10':>8}  {'P50':>8}  {'P90':>8}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, m in zip(axes, model_names):
        oof_p = oofs[m]
        test_p = test_preds[m]

        for split, arr, color in [("OOF", oof_p, "steelblue"), ("Test", test_p, "darkorange")]:
            p10, p50, p90 = np.percentile(arr, [10, 50, 90])
            print(
                f"  {m.upper():<12}  {split:<6}  {arr.mean():>8.4f}  {arr.std():>8.4f}"
                f"  {p10:>8.4f}  {p50:>8.4f}  {p90:>8.4f}"
            )
            # KDE via histogram
            ax.hist(arr, bins=80, density=True, alpha=0.5, color=color, label=split)

        ax.set_title(m.upper())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("OOF vs Test Prediction Distributions", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "test_pred_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT_DIR / 'test_pred_distributions.png'}")


# ---------------------------------------------------------------------------
# Section 6: Numeric marginal KS tests
# ---------------------------------------------------------------------------


def section6_ks_tests(train: pl.DataFrame, test: pl.DataFrame) -> None:
    _sep("Section 6 — Numeric Marginal KS Tests")

    available = [f for f in NUM_FEATURES if f in train.columns and f in test.columns]
    results = []
    for feat in available:
        tr_arr = train[feat].drop_nulls().to_numpy()
        te_arr = test[feat].drop_nulls().to_numpy()
        ks_stat, p_val = stats.ks_2samp(tr_arr, te_arr)
        results.append((feat, ks_stat, p_val))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<35}  {'KS stat':>10}  {'p-value':>12}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*12}")
    for feat, ks, p in results:
        flag = " ⚠" if ks > 0.05 else ""
        print(f"  {feat:<35}  {ks:>10.4f}  {p:>12.4e}{flag}")

    # KDE plot for top 5 most-shifted
    top5 = results[:5]
    fig, axes = plt.subplots(1, len(top5), figsize=(5 * len(top5), 4))
    if len(top5) == 1:
        axes = [axes]
    for ax, (feat, ks, p) in zip(axes, top5):
        tr_arr = train[feat].drop_nulls().to_numpy()
        te_arr = test[feat].drop_nulls().to_numpy()
        ax.hist(tr_arr, bins=60, density=True, alpha=0.5, color="steelblue", label="Train")
        ax.hist(te_arr, bins=60, density=True, alpha=0.5, color="darkorange", label="Test")
        ax.set_title(f"{feat}\nKS={ks:.3f}")
        ax.set_xlabel(feat)
        ax.legend()
    fig.suptitle("Top Numerically Shifted Features (Train vs Test)", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ks_features.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT_DIR / 'ks_features.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("PS S6E5 — Distribution Shift Analysis")
    print(f"Output directory: {OUT_DIR}")

    train, test = _load_data()

    # Section 1
    test_year_frac = section1_set_membership(train, test)

    # Section 2
    section2_adversarial(train, test)

    # Load sorted train + OOF arrays for sections 3–5
    train_sorted = build_features(train)
    y, year, oofs = _load_oof_arrays(train_sorted)
    test_preds = _load_test_arrays()

    # Section 3
    per_year_aucs = section3_per_year_oof(train_sorted, y, year, oofs)

    # Section 4
    section4_expected_lb(y, year, oofs, per_year_aucs, test_year_frac)

    # Section 5
    section5_pred_distributions(y, oofs, test_preds)

    # Section 6
    section6_ks_tests(train, test)

    print("\n\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
