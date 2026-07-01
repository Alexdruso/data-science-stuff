"""Reproducible figures for the S6E6 competition writeup.

Run: python playground-series-s6e6/notebooks/writeup_figures.py
Outputs three PNGs to results/analysis/.

All numbers are the verified final values (private LB revealed 2026-07-01):
  - 12th place (private) = the two selected metablend finals, both private 0.97041
  - best PUBLIC submission = metablend 0.97119 -> private 0.97033 (NOT selected)
  - gbdtstack 0.97094 public -> private 0.97043 (best private overall, not selected)
  - public standing = rank 407 / 2817, score 0.97119
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Muted, print-friendly palette.
INK = "#1f2937"
MUTED = "#9ca3af"
KEEP = "#2563eb"  # what we kept / 12th
TRAP = "#dc2626"  # the shiny public pick we skipped
OTHER = "#059669"  # gbdtstack

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig_shakeup() -> None:
    """Slope chart: public score -> private score for our key submissions."""
    # (label, public, private, colour, note)
    subs = [
        ("metablend (best public)", 0.97119, 0.97033, TRAP, "NOT selected"),
        ("metablend (selected finals)", 0.97108, 0.97041, KEEP, "12th place"),
        ("lrstack", 0.97105, 0.97039, MUTED, ""),
        ("gbdtstack", 0.97094, 0.97042, OTHER, "best private"),
    ]

    # small vertical nudges (in score units) for crowded left-side labels
    left_dy = {
        "metablend (best public)": 0.0,
        "metablend (selected finals)": 0.00006,
        "lrstack": -0.00006,
        "gbdtstack": 0.0,
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    x_pub, x_priv = 0.0, 1.0
    for label, pub, priv, colour, note in subs:
        lw = 3.0 if colour in (TRAP, KEEP) else 1.8
        alpha = 1.0 if colour in (TRAP, KEEP, OTHER) else 0.7
        ax.plot([x_pub, x_priv], [pub, priv], color=colour, lw=lw, alpha=alpha,
                marker="o", markersize=7, zorder=3)
        # left labels (public)
        ax.text(x_pub - 0.03, pub + left_dy[label], f"{label}  {pub:.5f}",
                ha="right", va="center", fontsize=9.5,
                color=colour if colour != MUTED else INK)
        # right labels (private)
        tag = f"{priv:.5f}"
        if note:
            tag += f"  ({note})"
        weight = "bold" if colour in (TRAP, KEEP) else "normal"
        ax.text(x_priv + 0.03, priv, tag, ha="left", va="center", fontsize=9.5,
                color=colour if colour != MUTED else INK, fontweight=weight)

    ax.set_xlim(-0.75, 1.7)
    ax.set_xticks([x_pub, x_priv])
    ax.set_xticklabels(["Public LB", "Private LB"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Public score near the top was anti-predictive")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#eef0f3", zorder=0)

    ax.annotate(
        "the pick that topped our\npublic LB finished lowest\non the private LB",
        xy=(x_priv - 0.02, 0.97034), xytext=(0.52, 0.97088),
        fontsize=9, color=TRAP, ha="left", va="center",
        arrowprops={"arrowstyle": "->", "color": TRAP, "lw": 1.2},
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_shakeup.png", bbox_inches="tight")
    plt.close(fig)


def fig_plateau_break() -> None:
    """The GBDT plateau and the diversity break."""
    # Ordered milestones of the best stack/model OOF balanced accuracy.
    steps = [
        ("baseline LGBM", 0.9559, "gbdt"),
        ("+ class_weight balanced", 0.9638, "gbdt"),
        ("tuned LGBM", 0.9657, "gbdt"),
        ("stack over GBDT cluster", 0.9662, "gbdt"),
        ("+ XGB 240-feat recipe", 0.9679, "div"),
        ("+ RealMLP (logit-adjust)", 0.9699, "div"),
        ("+ CatBoost + chain-cascade", 0.9705, "div"),
    ]
    labels = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, vals, color=INK, lw=1.6, zorder=2)
    for xi, (_, v, kind) in zip(x, steps):
        c = MUTED if kind == "gbdt" else KEEP
        ax.scatter(xi, v, color=c, s=70, zorder=3)

    # Plateau band + caption in open space below-right, pointing to the flat segment.
    ax.axhspan(0.9655, 0.9664, color="#eceef1", zorder=0)
    ax.annotate(
        "GBDT plateau:\nmore of the same family\nbarely moved (+0.0005)",
        xy=(3.0, 0.9662), xytext=(3.3, 0.9600),
        fontsize=9.5, color="#6b7280", ha="left", va="center",
        arrowprops={"arrowstyle": "->", "color": "#9ca3af", "lw": 1.2},
    )

    # Break arrow.
    ax.annotate(
        "diverse model families\nbreak the plateau",
        xy=(4.0, 0.9679), xytext=(2.1, 0.9702),
        fontsize=10, color=KEEP, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": KEEP, "lw": 1.6},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("OOF balanced accuracy (5-fold)")
    ax.set_title("What actually moved the needle: diversity, not tuning")
    ax.grid(axis="y", color="#eef0f3", zorder=0)
    ax.set_ylim(0.954, 0.9715)
    fig.tight_layout()
    fig.savefig(OUT / "fig_plateau_break.png", bbox_inches="tight")
    plt.close(fig)


def fig_lb_density() -> None:
    """Public LB score histogram near the top — why public LB was noise."""
    lb_path = Path("/tmp/s6e6_lb")
    csvs = list(lb_path.glob("*publicleaderboard*.csv"))
    if not csvs:
        print("SKIP fig_lb_density: leaderboard CSV not found in /tmp/s6e6_lb")
        return
    import csv as _csv

    with open(csvs[0], encoding="utf-8-sig") as fh:
        scores = np.array([float(r["Score"]) for r in _csv.DictReader(fh)])

    our_pub = 0.97119
    near = scores[scores >= 0.9695]
    within = int(np.sum(np.abs(scores - our_pub) < 0.0001))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.hist(near, bins=60, color="#c7d2fe", edgecolor="#a5b4fc", linewidth=0.4)
    ax.axvline(our_pub, color=KEEP, lw=2.2)
    ax.text(our_pub, ax.get_ylim()[1] * 0.92,
            f"  us (public 0.97119)\n  {within} teams within ±0.0001",
            color=KEEP, fontsize=9.5, va="top", fontweight="bold")
    ax.set_xlabel("Public balanced accuracy")
    ax.set_ylabel("Teams")
    ax.set_title("The top of the public LB was a knife-edge")
    ax.grid(axis="y", color="#eef0f3", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lb_density.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_shakeup()
    fig_plateau_break()
    fig_lb_density()
    print(f"Figures written to {OUT}")
