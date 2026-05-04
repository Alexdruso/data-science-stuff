# %% [markdown]
# # PS S6E5 — F1 Pit Stop Prediction: EDA
# Binary classification: predict whether a driver will pit on the next lap.
# Metric: AUC-ROC

# %% Imports & config
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
import seaborn as sns

DATA_DIR = Path(__file__).parent.parent / "data"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 100

TARGET = "PitNextLap"
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

# %% Load data
train = pl.read_csv(DATA_DIR / "train.csv")
test = pl.read_csv(DATA_DIR / "test.csv")

print(f"Train: {train.shape}  |  Test: {test.shape}")

# %% Overview — schema, nulls, descriptive stats
print("=== SCHEMA ===")
print(train.schema)

print("\n=== NULL COUNTS (train) ===")
null_counts = train.null_count()
print(null_counts)

print("\n=== DESCRIBE (numeric) ===")
print(train.select(NUM_FEATURES + [TARGET]).describe())

# %% Target distribution
target_counts = (
    train.group_by(TARGET).agg(pl.len().alias("count")).sort(TARGET).to_pandas()
)
total = target_counts["count"].sum()
target_counts["pct"] = target_counts["count"] / total * 100

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    target_counts[TARGET].astype(str),
    target_counts["count"],
    color=["#4878cf", "#d65f5f"],
)
for bar, (_, row) in zip(bars, target_counts.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{row['pct']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_xlabel("PitNextLap")
ax.set_ylabel("Count")
ax.set_title("Target Distribution")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.show()


# %% Categorical features — count + pit rate
def plot_cat_pit_rate(
    df: pl.DataFrame,
    col: str,
    top_n: int = 20,
    title: str | None = None,
) -> None:
    agg = (
        df.group_by(col)
        .agg(
            pl.len().alias("count"),
            pl.col(TARGET).mean().alias("pit_rate"),
        )
        .sort("count", descending=True)
        .head(top_n)
        .sort("pit_rate", descending=True)
        .to_pandas()
    )

    fig, ax1 = plt.subplots(figsize=(12, 4))
    x = range(len(agg))
    ax1.bar(x, agg["count"], color="#4878cf", alpha=0.7, label="Count")
    ax1.set_ylabel("Count", color="#4878cf")
    ax1.tick_params(axis="y", labelcolor="#4878cf")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(agg[col].astype(str), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        x, agg["pit_rate"], color="#d65f5f", marker="o", linewidth=2, label="Pit rate"
    )
    ax2.set_ylabel("Pit Rate", color="#d65f5f")
    ax2.tick_params(axis="y", labelcolor="#d65f5f")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    ax1.set_title(title or f"{col} — count & pit rate")
    fig.tight_layout()
    plt.show()


plot_cat_pit_rate(train, "Compound", top_n=4, title="Compound — count & pit rate")
plot_cat_pit_rate(train, "Year", top_n=10, title="Year — count & pit rate")
plot_cat_pit_rate(
    train, "Driver", top_n=20, title="Driver — top 20 by count & pit rate"
)
plot_cat_pit_rate(train, "Race", top_n=20, title="Race — top 20 by count & pit rate")

# %% Numeric feature distributions — by target
n_cols = 3
n_rows = -(-len(NUM_FEATURES) // n_cols)  # ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes_flat = axes.flatten()

train_pd = train.select(NUM_FEATURES + [TARGET]).to_pandas()

for i, feat in enumerate(NUM_FEATURES):
    ax = axes_flat[i]
    for val, color, label in [(0, "#4878cf", "No pit"), (1, "#d65f5f", "Pit")]:
        subset = train_pd.loc[train_pd[TARGET] == val, feat].dropna()
        ax.hist(subset, bins=40, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Numeric Feature Distributions by PitNextLap", y=1.01)
plt.tight_layout()
plt.show()

# %% PitStop flag — leakage investigation
print("=== PitStop × PitNextLap cross-tab ===")
crosstab = (
    train.group_by(["PitStop", TARGET])
    .agg(pl.len().alias("count"))
    .sort(["PitStop", TARGET])
)
print(crosstab)

pit_rate_by_pitstop = (
    train.group_by("PitStop")
    .agg(pl.col(TARGET).mean().alias("pit_next_lap_rate"), pl.len().alias("count"))
    .sort("PitStop")
    .to_pandas()
)
print("\nPit rate on NEXT lap conditioned on PitStop this lap:")
print(pit_rate_by_pitstop.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    pit_rate_by_pitstop["PitStop"].astype(str),
    pit_rate_by_pitstop["pit_next_lap_rate"],
    color=["#4878cf", "#d65f5f"],
)
ax.set_xlabel("PitStop (this lap)")
ax.set_ylabel("PitNextLap rate")
ax.set_title("Pit-on-next-lap rate conditioned on pitting this lap")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.tight_layout()
plt.show()


# %% TyreLife & RaceProgress vs pit rate
def plot_binned_pit_rate(
    df: pl.DataFrame,
    col: str,
    n_bins: int = 20,
) -> None:
    pdf = df.select([col, TARGET]).to_pandas()
    pdf["bin"] = (
        pl.Series((df[col] - df[col].min()) / (df[col].max() - df[col].min()) * n_bins)
        .cast(pl.Int32)
        .clip(0, n_bins - 1)
        .to_pandas()
    )

    binned = (
        pdf.groupby("bin", observed=True)
        .agg(
            pit_rate=(TARGET, "mean"),
            count=(TARGET, "count"),
            bin_mid=(col, "mean"),
        )
        .sort_values("bin")
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        binned["bin_mid"],
        binned["count"],
        width=(binned["bin_mid"].iloc[1] - binned["bin_mid"].iloc[0]) * 0.8
        if len(binned) > 1
        else 1,
        alpha=0.4,
        color="#4878cf",
        label="Count",
    )
    ax1.set_xlabel(col)
    ax1.set_ylabel("Count", color="#4878cf")
    ax1.tick_params(axis="y", labelcolor="#4878cf")

    ax2 = ax1.twinx()
    ax2.plot(
        binned["bin_mid"],
        binned["pit_rate"],
        color="#d65f5f",
        marker="o",
        linewidth=2,
        label="Pit rate",
    )
    ax2.set_ylabel("Pit Rate", color="#d65f5f")
    ax2.tick_params(axis="y", labelcolor="#d65f5f")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    ax1.set_title(f"{col} vs Pit Rate")
    fig.tight_layout()
    plt.show()


plot_binned_pit_rate(train, "TyreLife")
plot_binned_pit_rate(train, "RaceProgress")
plot_binned_pit_rate(train, "LapNumber")

# %% Correlation matrix
corr_cols = NUM_FEATURES + [TARGET, "PitStop"]
corr = train.select(corr_cols).to_pandas().corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = __import__("numpy").triu(__import__("numpy").ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    square=True,
    linewidths=0.5,
)
ax.set_title("Pearson Correlation Matrix (train)")
plt.tight_layout()
plt.show()


# %% Train vs Test distribution check
def plot_train_test_kde(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
) -> None:
    n_cols = 3
    n_rows = -(-len(features) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes_flat = axes.flatten()

    train_pd = train_df.select(features).to_pandas()
    test_pd = test_df.select(features).to_pandas()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        ax.hist(
            train_pd[feat].dropna(),
            bins=40,
            alpha=0.5,
            density=True,
            color="#4878cf",
            label="Train",
        )
        ax.hist(
            test_pd[feat].dropna(),
            bins=40,
            alpha=0.5,
            density=True,
            color="#d65f5f",
            label="Test",
        )
        ax.set_title(feat)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Train vs Test: Numeric Feature Distributions", y=1.01)
    plt.tight_layout()
    plt.show()


plot_train_test_kde(train, test, NUM_FEATURES)

# Categorical shift check
for col in ["Compound", "Year"]:
    train_vc = (
        train.group_by(col)
        .agg(pl.len().alias("train_count"))
        .with_columns(
            (pl.col("train_count") / pl.col("train_count").sum()).alias("train_pct")
        )
    )
    test_vc = (
        test.group_by(col)
        .agg(pl.len().alias("test_count"))
        .with_columns(
            (pl.col("test_count") / pl.col("test_count").sum()).alias("test_pct")
        )
    )
    joined = train_vc.join(test_vc, on=col, how="full").sort(col)
    print(f"\n{col} — train vs test proportions:")
    print(joined)

# %% [markdown]
# ## Key Insights
#
# - **Class imbalance**: `PitNextLap=1` is a minority class (~5-10% of laps).
#   Use stratified CV and monitor AUC not accuracy.
#
# - **PitStop flag (this lap)**: A driver who pits *this* lap almost certainly
#   will NOT pit next lap — investigate whether this creates leakage or is a
#   valid feature (it should be valid: strategy decisions are sequential).
#
# - **TyreLife is likely the strongest predictor**: Pit probability rises sharply
#   after ~20 laps on a tyre. Bin or polynomial features may help.
#
# - **RaceProgress**: Pit probability clusters in the first ~40% and ~70% of the
#   race (typical 2-stop strategy windows). Consider interaction with Compound.
#
# - **Compound × TyreLife interaction**: SOFT tyres degrade faster — an interaction
#   term `Compound_SOFT × TyreLife` may capture tyre-age sensitivity by compound.
#
# - **Position_Change & LapTime_Delta**: Near-zero distributions skewed by outliers.
#   Clipping or robust scaling may improve tree splits.
#
# - **Train/Test shift**: Check for year or circuit distribution differences —
#   if test skews toward recent years, year-based features need careful encoding.
