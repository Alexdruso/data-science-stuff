"""SHAP-based feature importance analysis for PS S6E5."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from lightgbm import LGBMClassifier
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import DRIVER_COLS, build_features, compute_group_features, compute_race_lap_features, compute_race_stint_features

from data_science_stuff.kaggle.io import competition_dirs, load_params

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
TARGET = "PitNextLap"

LGBM_DEFAULTS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
    "device": "gpu",
}

def main() -> None:
    print("Loading data and building features...")
    train_raw = pl.read_csv(DATA_DIR / "train.csv")

    train_pl = build_features(train_raw)
    train_pl = compute_group_features(train_raw, train_pl)
    train_pl = compute_race_lap_features(train_pl)
    train_pl = compute_race_stint_features(train_raw, train_pl)
    print(f"Feature matrix: {train_pl.shape}")

    _exclude = {"id", TARGET} | DRIVER_COLS
    cat_cols = [
        c for c in train_pl.columns
        if train_pl[c].dtype == pl.String and c not in _exclude
    ]
    feature_cols = [c for c in train_pl.columns if c not in _exclude]

    train = train_pl.to_pandas()
    for col in cat_cols:
        train[col] = train[col].astype("category")

    X = train[feature_cols].copy()
    y = train[TARGET].to_numpy()

    # Full-data driver target encoding (no fold needed for exploration)
    driver_arr = train_pl["Driver"].to_numpy()
    te = TargetEncoder(smooth="auto", random_state=42)
    X["driver_target_enc"] = te.fit_transform(driver_arr.reshape(-1, 1), y).ravel()

    # --- Train LGBM on full dataset (GPU) ---
    print("Training LGBM on full dataset (GPU, 500 trees)...")
    params = load_params(RESULTS_DIR, LGBM_DEFAULTS, "best_params.json")
    params["n_estimators"] = 500  # fixed for exploration
    params.pop("early_stopping_rounds", None)
    model = LGBMClassifier(**params)
    model.fit(X, y, categorical_feature=cat_cols)
    print("Done.")

    # --- Stratified sample: 2000 rows per year ---
    year_col = train["Year"].values
    rng = np.random.default_rng(42)
    sample_idx_list = []
    for yr in sorted(np.unique(year_col)):
        mask = np.where(year_col == yr)[0]
        n = min(2000, len(mask))
        chosen = rng.choice(mask, size=n, replace=False)
        sample_idx_list.append(chosen)
    sample_idx = np.concatenate(sample_idx_list)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    year_sample = year_col[sample_idx]
    print(f"SHAP sample: {len(X_sample)} rows (2000/year × {len(np.unique(year_col))} years)")

    # --- SHAP values ---
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    # shap_vals may be (n, f) or list[(n,f),(n,f)] depending on shap version
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    print("Done.")

    feature_names = list(X.columns)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # LGBM native importances
    gain_imp = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "lgbm_gain": gain_imp,
            "lgbm_split": split_imp,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR / 'shap_importance.csv'}")

    # --- Console: top 20 + bottom 10 ---
    print("\n" + "="*60)
    print("TOP 20 FEATURES (mean |SHAP|)")
    print("="*60)
    print(importance_df.head(20).to_string(index=False))

    bottom = importance_df[importance_df["mean_abs_shap"] < 1e-4]
    print(f"\n{'='*60}")
    print(f"NEAR-ZERO SHAP (< 1e-4): {len(bottom)} features")
    print("="*60)
    if len(bottom):
        print(bottom["feature"].tolist())
    else:
        print("None — all features have measurable SHAP contribution.")

    # --- Correlated feature pairs ---
    print(f"\n{'='*60}")
    print("HIGHLY CORRELATED PAIRS (|r| > 0.85)")
    print("="*60)
    num_cols = [c for c in feature_names if X[c].dtype != "category"]
    corr = X[num_cols].corr().abs()
    pairs = []
    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if j <= i:
                continue
            r = corr.loc[a, b]
            if r > 0.85:
                pairs.append((a, b, round(r, 3)))
    if pairs:
        pairs.sort(key=lambda t: -t[2])
        for a, b, r in pairs:
            print(f"  {a:40s}  ↔  {b:40s}  r={r}")
    else:
        print("No pairs above threshold.")

    # --- Per-year top-5 SHAP ---
    print(f"\n{'='*60}")
    print("TOP 5 FEATURES BY YEAR")
    print("="*60)
    for yr in sorted(np.unique(year_sample)):
        mask = year_sample == yr
        yr_mean = np.abs(shap_vals[mask]).mean(axis=0)
        top5_idx = np.argsort(yr_mean)[::-1][:5]
        top5 = [(feature_names[i], round(yr_mean[i], 5)) for i in top5_idx]
        print(f"\n  Year {yr} (n={mask.sum()}):")
        for feat, val in top5:
            print(f"    {feat:40s}  {val:.5f}")

    # --- Plots ---
    print(f"\n{'='*60}")
    print("Saving plots...")

    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved → {RESULTS_DIR / 'shap_summary.png'}")
    print(f"Saved → {RESULTS_DIR / 'shap_bar.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
