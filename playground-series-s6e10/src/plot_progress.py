"""Plot how the best honest OOF AUC evolved over the s6e10 campaign.

Milestones are transcribed from CLAUDE.md ("Current best", submission log and the dated
session notes). Writes `figures/score_progress.svg` (SVG because `*.png` is gitignored).

    python src/plot_progress.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = "#2a78d6"

# (date, best honest OOF AUC, label) — each row is a new campaign best.
MILESTONES: list[tuple[date, float, str]] = [
    (date(2026, 10, 5), 0.95874, "LGBM baseline"),
    (date(2026, 10, 5), 0.959169, "tuned LGBM + RealMLP"),
    (date(2026, 10, 7), 0.960932, "flight distance as route ID"),
    (date(2026, 10, 7), 0.961435, "route TE + embeddings + original rows"),
    (date(2026, 10, 8), 0.961695, "teacher logit + aux-task legs"),
    (date(2026, 10, 9), 0.962162, "LR stack over public OOF pool"),
    (date(2026, 10, 10), 0.962196, "bagged stack"),
    (date(2026, 10, 15), 0.962238, "strong-C LR, all own legs (final)"),
]
DEADLINE = date(2026, 10, 31)

# Label placement offsets in points (dx, dy), tuned so labels don't collide.
OFFSETS: list[tuple[float, float]] = [
    (8, -4),
    (8, -4),
    (8, -10),
    (8, -8),
    (8, -10),
    (10, -16),
    (-6, 8),
    (8, 8),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "figures" / "score_progress.svg"
    out.parent.mkdir(exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    days = [m[0] for m in MILESTONES]
    aucs = [m[1] for m in MILESTONES]

    # Step line: the best stays in force until the next one lands; carry to deadline.
    ax.step(
        [*days, DEADLINE],
        [*aucs, aucs[-1]],
        where="post",
        color=SERIES,
        linewidth=2,
        solid_joinstyle="round",
    )
    ax.scatter(
        days, aucs, s=50, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3
    )

    for (d, auc, label), (dx, dy) in zip(MILESTONES, OFFSETS):
        ax.annotate(
            f"{label}  {auc:.6f}",
            (d, auc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            ha="left" if dx > 0 else "right",
            va="top" if dy < 0 else "bottom",
        )

    ax.annotate(
        "10-16 → 10-31: 17 more pre-registered levers (PR-43 to PR-58), all null",
        (date(2026, 10, 23), aucs[-1]),
        xytext=(0, -14),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
        ha="center",
        va="top",
    )

    ax.set_title(
        "Playground S6E10 — best honest OOF ROC AUC over the campaign",
        loc="left",
        fontsize=12,
        color=INK,
        pad=12,
    )
    ax.set_ylabel("OOF ROC AUC", color=INK_2)
    ax.set_xlim(date(2026, 10, 4), date(2026, 11, 1))
    ax.set_ylim(0.9584, 0.9626)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[5, 10, 15, 20, 25, 31]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
