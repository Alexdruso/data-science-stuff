"""EDA for PS S6E6 — Predicting Stellar Class.

Run from the competition root:
    python notebooks/eda.py

Outputs go to results/analysis/ (created automatically).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "class"


def main() -> None:
    train = pl.read_csv(DATA_DIR / "train.csv")
    test = pl.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {train.shape}   Test: {test.shape}")

    # ── class distribution ─────────────────────────────────────────────────
    print("\nClass distribution (train):")
    print(train[TARGET].value_counts(sort=True))

    # ── missing values ─────────────────────────────────────────────────────
    null_counts = train.null_count()
    print("\nNull counts (train):")
    print(null_counts)

    # ── column overlap train / test ────────────────────────────────────────
    train_cols = set(train.columns) - {TARGET}
    test_cols = set(test.columns)
    if train_cols != test_cols:
        print(f"\n⚠️  Column mismatch — train only: {train_cols - test_cols}, test only: {test_cols - train_cols}")
    else:
        print("\nColumn overlap: train features == test features ✓")

    # ── numeric summaries ──────────────────────────────────────────────────
    print("\nTrain describe:")
    print(train.describe())

    # ── per-feature histograms (numeric columns) ───────────────────────────
    numeric_cols = [c for c, t in zip(train.columns, train.dtypes) if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in {"id"}]
    n_cols = 3
    n_rows = -(-len(numeric_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    ax_flat = [ax for row in (axes if n_rows > 1 else [axes]) for ax in (row if hasattr(row, "__iter__") else [row])]

    for ax, col in zip(ax_flat, numeric_cols):
        data = train[col].drop_nulls().to_numpy()
        ax.hist(data, bins=50, edgecolor="none", alpha=0.7)
        ax.set_title(col, fontsize=8)
        ax.set_yticks([])

    for ax in ax_flat[len(numeric_cols):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "feature_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nHistograms saved → {out}")

    # ── class-conditional means ────────────────────────────────────────────
    print("\nClass-conditional feature means:")
    print(train.group_by(TARGET).agg([pl.col(c).mean() for c in numeric_cols]).sort(TARGET))


if __name__ == "__main__":
    main()
