"""EDA: does `id` mean anything? (ordinality / row-order artifacts)

Stages:
  A  id structure — contiguity, gaps, train/test boundary
  B  target vs id — class rates per id bin + chi-square on deciles
  C  target autocorrelation in id order — lag-k agreement vs shuffled null
  D  id-as-feature — LGBM on id (+ modulo features) alone, 5-fold balanced acc
  E  feature drift over id — per-bin z-scores of numeric means, train + test
  F  missingness masks vs id — mask rates per id bin, train and test

Report -> results/eda_id.txt, figures -> results/figures/id_*.png
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# dataviz palette
C1, C2, C3, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CLASSES = ["at-risk", "unhealthy", "fit"]
CLS_COLOR = {"at-risk": C1, "unhealthy": RED, "fit": C2}
NUM = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CAT = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
MASK_COLS = [
    "water_intake",
    "physical_activity_level",
    "bmi",
    "diet_type",
    "stress_level",
    "sleep_duration",
]


def stage_a(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    print("=" * 70)
    print("A. id structure")
    print("=" * 70)
    for name, df in [("train", tr), ("test", te)]:
        ids = df["id"].to_numpy()
        contiguous = bool((np.diff(np.sort(ids)) == 1).all())
        sorted_already = bool((np.diff(ids) > 0).all())
        print(
            f"{name}: n={len(ids):,}  min={ids.min()}  max={ids.max()}  "
            f"contiguous={contiguous}  file-sorted={sorted_already}"
        )
    print(f"boundary: train max {tr['id'].max()} -> test min {te['id'].min()}")


def stage_b(tr: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("B. class rates vs id (100 bins) + chi-square on deciles")
    print("=" * 70)
    nbins = 100
    tr = tr.sort_values("id")
    binidx = pd.qcut(tr["id"], nbins, labels=False)
    rates = (
        pd.crosstab(binidx, tr["health_condition"], normalize="index")
        .reindex(columns=CLASSES)
    )
    overall = tr["health_condition"].value_counts(normalize=True)

    dec = pd.qcut(tr["id"], 10, labels=False)
    ct = pd.crosstab(dec, tr["health_condition"])
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"chi2(class x id-decile) = {chi2:.1f}, dof={dof}, p = {p:.3g}")
    for c in CLASSES:
        r = rates[c]
        print(
            f"  {c:<10} overall={overall[c]:.4f}  bin-rate range "
            f"[{r.min():.4f}, {r.max():.4f}]  sd={r.std():.5f}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, c in zip(axes, CLASSES):
        ax.plot(rates.index, rates[c], color=CLS_COLOR[c], lw=1.4)
        ax.axhline(overall[c], color=INK2, lw=0.8, ls="--")
        ax.set_ylabel(c, color=INK2)
    axes[-1].set_xlabel("id bin (train, 100 equal-count bins)")
    fig.suptitle(
        "Class rate per id bin — dashed = overall rate", color=INK, y=0.99
    )
    fig.tight_layout()
    fig.savefig(FIGS / "id_class_rates.png", dpi=130)
    plt.close(fig)
    print("fig -> id_class_rates.png")


def stage_c(tr: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("C. target autocorrelation in id order (lag-k agreement)")
    print("=" * 70)
    y = tr.sort_values("id")["health_condition"].to_numpy()
    p = pd.Series(y).value_counts(normalize=True)
    null = float((p**2).sum())
    n = len(y)
    lags = list(range(1, 11))
    agree = []
    print(f"shuffled-null agreement = sum p_k^2 = {null:.5f}")
    for k in lags:
        a = float((y[:-k] == y[k:]).mean())
        # binomial sd under null
        sd = float(np.sqrt(null * (1 - null) / (n - k)))
        z = (a - null) / sd
        agree.append(a)
        print(f"  lag {k:>2}: agreement {a:.5f}  (z vs null = {z:+.2f})")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(lags, agree, color=C1, width=0.7)
    ax.axhline(null, color=RED, lw=1.2, ls="--", label=f"shuffled null {null:.4f}")
    lo = min(min(agree), null)
    hi = max(max(agree), null)
    pad = 5 * (hi - lo + 1e-5)
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlabel("lag k (rows apart in id order)")
    ax.set_ylabel("P(y_i = y_{i+k})")
    ax.set_title("Adjacent-row target agreement vs shuffled null", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "id_lag_agreement.png", dpi=130)
    plt.close(fig)
    print("fig -> id_lag_agreement.png")


def stage_d(tr: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("D. id-as-feature: LGBM on id + modulo features, 5-fold bal-acc")
    print("=" * 70)
    from lightgbm import LGBMClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    X = pd.DataFrame(
        {
            "id": tr["id"],
            "id_mod2": tr["id"] % 2,
            "id_mod10": tr["id"] % 10,
            "id_mod100": tr["id"] % 100,
            "id_mod1000": tr["id"] % 1000,
        }
    )
    y = tr["health_condition"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baccs = []
    for fold, (itr, iva) in enumerate(skf.split(X, y)):
        m = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.1,
            class_weight="balanced",
            verbose=-1,
            random_state=42,
        )
        m.fit(X.iloc[itr], y[itr])
        pred = m.predict(X.iloc[iva])
        b = balanced_accuracy_score(y[iva], pred)
        baccs.append(b)
        print(f"  fold {fold}: bal-acc {b:.5f}")
    print(
        f"mean bal-acc = {np.mean(baccs):.5f}  "
        f"(chance = 0.33333; signal iff clearly above)"
    )


def stage_e(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E. numeric-feature drift over id (train bins + test bins)")
    print("=" * 70)
    nbins_tr, nbins_te = 50, 20
    tr = tr.sort_values("id")
    te = te.sort_values("id")
    btr = pd.qcut(tr["id"], nbins_tr, labels=False)
    bte = pd.qcut(te["id"], nbins_te, labels=False) + nbins_tr

    rows = []
    for c in NUM:
        mtr = tr.groupby(btr)[c].mean()
        mte = te.groupby(bte)[c].mean()
        allm = pd.concat([mtr, mte])
        mu, sd = tr[c].mean(), tr[c].std()
        se = sd / np.sqrt(len(tr) / nbins_tr)  # per-train-bin standard error
        z = (allm - mu) / se
        rows.append(z.rename(c))
        zmax_tr = float(z.iloc[:nbins_tr].abs().max())
        zte = float(z.iloc[nbins_tr:].mean())
        print(
            f"  {c:<22} max|z| within train bins = {zmax_tr:5.2f}   "
            f"mean z of test bins = {zte:+6.2f}"
        )
    Z = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    vmax = max(3.0, float(np.abs(Z.to_numpy()).max()))
    im = ax.imshow(
        Z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.axvline(nbins_tr - 0.5, color=INK, lw=1.4)
    ax.text(
        nbins_tr - 1.5,
        len(NUM) - 0.6,
        "train | test",
        color=INK,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(NUM)), NUM)
    ax.set_xlabel("id bin (train 50 bins, then test 20 bins)")
    ax.set_title(
        "Bin-mean z-score vs train mean (per-bin SE units)", color=INK
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="z")
    fig.tight_layout()
    fig.savefig(FIGS / "id_feature_drift.png", dpi=130)
    plt.close(fig)
    print("fig -> id_feature_drift.png")


def stage_f(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("F. missingness-mask rates vs id (train + test)")
    print("=" * 70)
    nbins_tr, nbins_te = 50, 20
    tr = tr.sort_values("id")
    te = te.sort_values("id")
    btr = pd.qcut(tr["id"], nbins_tr, labels=False)
    bte = pd.qcut(te["id"], nbins_te, labels=False)

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex="col")
    for i, c in enumerate(MASK_COLS):
        ax = axes[i // 2][i % 2]
        rtr = tr[c].isna().groupby(btr).mean()
        rte = te[c].isna().groupby(bte).mean()
        ax.plot(
            np.linspace(0, 1, nbins_tr), rtr, color=C1, lw=1.3, label="train"
        )
        ax.plot(
            np.linspace(0, 1, nbins_te), rte, color=C2, lw=1.3, label="test"
        )
        ax.set_title(c, color=INK, fontsize=10)
        sd_tr, sd_te = float(rtr.std()), float(rte.std())
        print(
            f"  {c:<22} train rate {tr[c].isna().mean():.4f} "
            f"(bin sd {sd_tr:.4f})   test rate {te[c].isna().mean():.4f} "
            f"(bin sd {sd_te:.4f})"
        )
    axes[0][0].legend(frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("relative position in id range")
    fig.suptitle("Missing rate per id bin — flat = no id dependence", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "id_mask_rates.png", dpi=130)
    plt.close(fig)
    print("fig -> id_mask_rates.png")


def stage_g(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("G. mask-regime blocks: rolling missing rate (w=4000), train vs test")
    print("=" * 70)
    w = 4000
    tr = tr.sort_values("id").reset_index(drop=True)
    te = te.sort_values("id").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=False)
    for ax, col in zip(axes, ["water_intake", "diet_type"]):
        rtr = tr[col].isna().astype(int).rolling(w, center=True).mean()
        rte = te[col].isna().astype(int).rolling(w, center=True).mean()
        ax.plot(
            np.linspace(0, 1, len(rtr)), rtr, color=C1, lw=1.0, label="train"
        )
        ax.plot(
            np.linspace(0, 1, len(rte)), rte, color=C2, lw=1.0, label="test"
        )
        ax.set_ylabel(f"{col}\nrolling missing rate", color=INK2, fontsize=9)
        zte = rte.dropna() < 0.001
        print(
            f"  {col:<14} test zero-regime share {zte.mean():.3f}, "
            f"active rate {rte.dropna()[~zte.to_numpy()].mean():.4f}; "
            f"train range [{rtr.min():.4f}, {rtr.max():.4f}]"
        )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("relative position in id range")
    fig.suptitle(
        "Test masks are blockwise in id (regimes incl. hard zero);"
        " train is stationary",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "id_mask_regimes.png", dpi=130)
    plt.close(fig)
    print("fig -> id_mask_regimes.png")


def main() -> None:
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s: str) -> int:  # noqa: D102
            buf.write(s)
            import sys

            sys.__stdout__.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        stage_a(tr, te)
        stage_b(tr)
        stage_c(tr)
        stage_d(tr)
        stage_e(tr, te)
        stage_f(tr, te)
        stage_g(tr, te)

    out = ROOT / "results" / "eda_id.txt"
    out.write_text(buf.getvalue())
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
