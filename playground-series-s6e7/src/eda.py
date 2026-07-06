"""EDA for PS S6E7 — Predicting Student Health Risk.

Prints and writes results/eda_summary.md:
- class balance, per-column null rates (train vs test),
- numeric summary + class-conditional means, categorical value shares by class,
- missingness-vs-target signal,
- adversarial validation (train-vs-test) AUC + top discriminating features.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import CAT_COLS, CLASSES, NUM_COLS, TARGET, build_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    n_tr, n_te = train.height, test.height
    lines: list[str] = ["# S6E7 EDA summary\n"]

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"train: {train.shape}   test: {test.shape}\n")

    # ── class balance ────────────────────────────────────────────────────────
    emit("## Class balance")
    vc = train[TARGET].value_counts(sort=True)
    for row in vc.iter_rows(named=True):
        emit(f"- {row[TARGET]}: {row['count']} ({_fmt_pct(row['count'] / n_tr)})")
    emit("")

    # ── null rates train vs test ─────────────────────────────────────────────
    emit("## Null rate (train / test)")
    feat_cols = NUM_COLS + CAT_COLS
    for c in feat_cols:
        emit(
            f"- {c}: {_fmt_pct(train[c].null_count() / n_tr)} / "
            f"{_fmt_pct(test[c].null_count() / n_te)}"
        )
    emit("")

    # ── numeric: overall + class-conditional means, train vs test mean ────────
    emit("## Numeric features — mean by class (train) | train mean / test mean")
    for c in NUM_COLS:
        by_class = " ".join(
            f"{cls}={train.filter(pl.col(TARGET) == cls)[c].mean():.2f}"
            for cls in CLASSES
        )
        emit(f"- {c}: {by_class} | {train[c].mean():.2f} / {test[c].mean():.2f}")
    emit("")

    # ── categorical value shares by class ────────────────────────────────────
    emit("## Categorical — value share within each class (train)")
    for c in CAT_COLS:
        emit(f"### {c}  (train share / test share)")
        vals = [v for v in train[c].unique().to_list() if v is not None]
        for v in sorted(vals):
            tr_share = _fmt_pct((train[c] == v).sum() / n_tr)
            te_share = _fmt_pct((test[c] == v).sum() / n_te)
            per_cls = " ".join(
                f"{cls}={_fmt_pct(((train[c] == v) & (train[TARGET] == cls)).sum() / max((train[TARGET] == cls).sum(), 1))}"
                for cls in CLASSES
            )
            emit(f"- {v}: {tr_share} / {te_share} | within-class: {per_cls}")
    emit("")

    # ── missingness vs target: null rate per class ───────────────────────────
    emit("## Missingness vs target — null rate of each feature within each class")
    for c in feat_cols:
        per_cls = " ".join(
            f"{cls}={_fmt_pct(train.filter(pl.col(TARGET) == cls)[c].null_count() / max((train[TARGET] == cls).sum(), 1))}"
            for cls in CLASSES
        )
        emit(f"- {c}: {per_cls}")
    emit("")

    # ── adversarial validation (train vs test) ───────────────────────────────
    emit("## Adversarial validation (train=0 vs test=1)")
    all_cols = NUM_COLS + CAT_COLS
    both = pl.concat([train.select(all_cols), test.select(all_cols)], how="vertical")
    is_test = np.concatenate([np.zeros(n_tr), np.ones(n_te)]).astype(np.int64)
    Xa = both.to_pandas()
    for c in CAT_COLS:
        Xa[c] = Xa[c].astype("category")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(Xa))
    for tr_idx, val_idx in skf.split(Xa, is_test):
        m = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        m.fit(Xa.iloc[tr_idx], is_test[tr_idx])
        oof[val_idx] = m.predict_proba(Xa.iloc[val_idx])[:, 1]
    adv_auc = roc_auc_score(is_test, oof)
    emit(
        f"- adversarial AUC: {adv_auc:.4f}  "
        f"({'~0.5 → same distribution, trust CV' if adv_auc < 0.55 else 'SHIFT — inspect top features'})"
    )

    # feature importance from a full-data fit
    m = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    m.fit(Xa, is_test)
    imp = sorted(zip(all_cols, m.feature_importances_), key=lambda t: -t[1])
    emit("- top discriminating features: " + ", ".join(f"{k}({v})" for k, v in imp[:8]))
    emit("")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "eda_summary.md"
    out.write_text("\n".join(lines))
    print(f"\nEDA summary written → {out}")


if __name__ == "__main__":
    main()
