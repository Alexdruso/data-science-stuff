"""SHAP + error analysis for PS S6E5 — LightGBM best model (v5)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
TARGET = "PitNextLap"
N_FOLDS = 5
FIXED_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def load_params() -> dict[str, object]:
    params: dict[str, object] = dict(FIXED_PARAMS)
    params_path = RESULTS_DIR / "best_params.json"
    if params_path.exists():
        with params_path.open() as f:
            params.update(json.load(f))
    return params


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train = build_features(train_raw)
    train = compute_group_features(train_raw, train)

    cat_cols = [
        c
        for c in train.columns
        if train[c].dtype == pl.String and c not in ("id", TARGET)
    ]
    feature_cols = [c for c in train.columns if c not in ("id", TARGET)]

    pdf = train.to_pandas()
    for col in cat_cols:
        pdf[col] = pdf[col].astype("category")

    X = pdf[feature_cols]
    y = pdf[TARGET].to_numpy()
    train_raw_pd = train_raw.to_pandas()
    return X, y, feature_cols, train_raw_pd


def run_cv_with_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    shap_vals = np.zeros(X.shape)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"  Fold {fold} AUC: {roc_auc_score(y_val, oof[val_idx]):.4f}")

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_vals[val_idx] = sv

    print(f"\nOOF AUC: {roc_auc_score(y, oof):.4f}")
    return oof, shap_vals


def plot_shap_importance(shap_vals: np.ndarray, feature_names: list[str]) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_n = 20

    print("\n=== SHAP Feature Importance (top 20) ===")
    for i in range(min(top_n, len(feature_names))):
        idx = order[i]
        print(f"  {feature_names[idx]:45s} {mean_abs[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [feature_names[i] for i in order[:top_n]][::-1]
    values = mean_abs[order[:top_n]][::-1]
    ax.barh(labels, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (top 20)")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "shap_importance.png", dpi=120)
    plt.close(fig)
    print(f"Saved → {ANALYSIS_DIR / 'shap_importance.png'}")


def plot_calibration(y: np.ndarray, oof: np.ndarray) -> None:
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10)

    print("\n=== Calibration (10 bins) ===")
    print(f"  {'Pred prob':>12}  {'Actual rate':>12}")
    for mp, fp in zip(mean_pred, frac_pos, strict=False):
        print(f"  {mp:12.3f}  {fp:12.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "calibration.png", dpi=120)
    plt.close(fig)
    print(f"Saved → {ANALYSIS_DIR / 'calibration.png'}")


def error_by_group(X: pd.DataFrame, y: np.ndarray, oof: np.ndarray) -> None:
    df = X.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    df["y"] = y
    df["pred"] = oof

    print("\n=== AUC by Year ===")
    for yr, grp in df.groupby("Year"):
        if grp["y"].nunique() == 2:
            auc = roc_auc_score(grp["y"], grp["pred"])
            print(
                f"  Year {yr}: AUC={auc:.4f}  pit_rate={grp['y'].mean():.3f}  n={len(grp)}"
            )

    print("\n=== AUC by Compound ===")
    for cmp, grp in df.groupby("Compound"):
        if grp["y"].nunique() == 2:
            auc = roc_auc_score(grp["y"], grp["pred"])
            print(
                f"  {str(cmp):14s}: AUC={auc:.4f}  pit_rate={grp['y'].mean():.3f}  n={len(grp)}"
            )

    print("\n=== AUC by TyreLife decile ===")
    df["tyre_decile"] = pd.qcut(df["TyreLife"], q=10, labels=False, duplicates="drop")
    for dec, grp in df.groupby("tyre_decile"):
        if grp["y"].nunique() == 2:
            auc = roc_auc_score(grp["y"], grp["pred"])
            tyre_range = f"{grp['TyreLife'].min():.0f}-{grp['TyreLife'].max():.0f}"
            print(
                f"  Decile {dec} (TyreLife {tyre_range:>8}): AUC={auc:.4f}"
                f"  pit_rate={grp['y'].mean():.3f}  n={len(grp)}"
            )

    print("\n=== Hardest Races (worst 15 by AUC) ===")
    race_results = []
    for race, grp in df.groupby("Race"):
        if grp["y"].nunique() == 2 and len(grp) >= 50:
            auc = roc_auc_score(grp["y"], grp["pred"])
            race_results.append((str(race), auc, grp["y"].mean(), len(grp)))
    race_results.sort(key=lambda x: x[1])
    for race, auc, pit_rate, n in race_results[:15]:
        print(f"  {race:35s}: AUC={auc:.4f}  pit_rate={pit_rate:.3f}  n={n}")


def temporal_analysis(
    X: pd.DataFrame,
    y: np.ndarray,
    oof: np.ndarray,
    train_raw: pd.DataFrame,
) -> None:
    df = train_raw[["Year", "Race", "Driver", "LapNumber", TARGET]].copy()
    df = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)

    df["pit_lap"] = np.where(
        df[TARGET] == 1, df["LapNumber"].astype(float), np.nan
    )
    df["next_pit_lap"] = df.groupby(["Year", "Race", "Driver"])["pit_lap"].transform(
        lambda x: x.bfill()
    )
    df["laps_to_next_pit"] = (
        (df["next_pit_lap"] - df["LapNumber"]).clip(0, 10).fillna(10)
    ).astype(int)

    X_al = X[["Year", "Race", "Driver", "LapNumber"]].copy()
    for col in ["Race", "Driver"]:
        if X_al[col].dtype.name == "category":
            X_al[col] = X_al[col].astype(str)
    X_al = X_al.reset_index(drop=True)
    X_al["y"] = y
    X_al["pred"] = oof

    lookup = df[["Year", "Race", "Driver", "LapNumber", "laps_to_next_pit"]]
    merged = X_al.merge(lookup, on=["Year", "Race", "Driver", "LapNumber"], how="left")
    merged["laps_to_next_pit"] = merged["laps_to_next_pit"].fillna(10).astype(int)

    print("\n=== AUC by Laps to Next Pit ===")
    bucket_results = []
    for laps, grp in merged.groupby("laps_to_next_pit"):
        if grp["y"].nunique() == 2:
            auc = roc_auc_score(grp["y"], grp["pred"])
            bucket_results.append((int(laps), auc, grp["y"].mean(), len(grp)))
            print(
                f"  laps_to_next_pit={laps:2d}: AUC={auc:.4f}"
                f"  pit_rate={grp['y'].mean():.3f}  n={len(grp)}"
            )

    laps_vals = [r[0] for r in bucket_results]
    auc_vals = [r[1] for r in bucket_results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(laps_vals, auc_vals, "o-")
    ax.set_xlabel("Laps to next pit stop (0 = this is the pit lap)")
    ax.set_ylabel("AUC")
    ax.set_title("Model AUC by temporal proximity to pit stop")
    ax.set_xticks(laps_vals)
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "temporal.png", dpi=120)
    plt.close(fig)
    print(f"Saved → {ANALYSIS_DIR / 'temporal.png'}")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()

    print("Preparing data...")
    X, y, feature_cols, train_raw = prepare_data()
    print(f"  Shape: {X.shape}")

    print("\nRunning 5-fold CV + SHAP...")
    oof, shap_vals = run_cv_with_shap(X, y, params)

    plot_shap_importance(shap_vals, feature_cols)
    plot_calibration(y, oof)
    error_by_group(X, y, oof)
    temporal_analysis(X, y, oof, train_raw)

    print("\nDone.")


if __name__ == "__main__":
    main()
