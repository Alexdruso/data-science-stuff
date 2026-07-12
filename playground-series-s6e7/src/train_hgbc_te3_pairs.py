"""Pairwise co-occurrence TE on the generator's ADDED dependence (Day-10, GF #1).

Day-9's fingerprint: the original's features are ~independent (|corr| 0.003)
but the generator ADDED ~0.10 cross-feature dependence, shared train/test.
Univariate te13 cannot represent P(y | v_i, v_j) and the 33-leaf regularized
config prevents the trees from assembling it from splits. Here: rank all 78
column pairs by |CramersV_synth - CramersV_original| (numerics decile-binned,
label-free), EB-TE the top PAIR-KEY strings alongside the standard te13
block, same tuned HGBC — paired vs hgbc_te3_u_s42 (0.9504).

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_pairs.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import NUM_COLS
from train_common import DATA_DIR, N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

EXT_CSV = DATA_DIR / "external" / "student_health_dataset_50k.csv"
N_PAIRS = 15
N_QUANT = 10


def binned_view(df: pd.DataFrame, cols: list[str], edges: dict) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c in edges:
            b = np.digitize(df[c].to_numpy(dtype=float), edges[c]).astype(float)
            b[df[c].isna().to_numpy()] = -1
            out[c] = pd.Series(b, index=df.index).astype(int).astype(str)
        else:
            out[c] = df[c].astype(str)
    return pd.DataFrame(out)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct.to_numpy())[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ds = load_dataset()
    raw_cols = ds.feature_cols
    ext = pd.read_csv(EXT_CSV)

    num_cols = [c for c in NUM_COLS if c in raw_cols]
    edges = {
        c: np.unique(np.nanquantile(ds.train[c], np.linspace(0, 1, N_QUANT + 1)[1:-1]))
        for c in num_cols
    }
    bin_tr = binned_view(ds.train, raw_cols, edges)
    bin_te = binned_view(ds.test, raw_cols, edges)
    bin_ext = binned_view(ext, raw_cols, edges)

    ranked = []
    for a, b in combinations(raw_cols, 2):
        dv = abs(cramers_v(bin_tr[a], bin_tr[b]) - cramers_v(bin_ext[a], bin_ext[b]))
        ranked.append((dv, a, b))
    ranked.sort(reverse=True)
    pairs = [(a, b) for _, a, b in ranked[:N_PAIRS]]
    print("top added-dependence pairs (|dV|):")
    for dv, a, b in ranked[:N_PAIRS]:
        print(f"  {a} x {b}: {dv:.4f}")

    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)

    te_tr_view = X[raw_cols].astype(str)
    te_te_view = X_test[raw_cols].astype(str)
    for a, b in pairs:
        te_tr_view[f"p_{a}__{b}"] = bin_tr[a] + "|" + bin_tr[b]
        te_te_view[f"p_{a}__{b}"] = bin_te[a] + "|" + bin_te[b]
    te_cols = list(te_tr_view.columns)
    te_names = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASSES)]
    print(f"TE over {len(te_cols)} cols ({len(te_names)} features), tuned HGBC")

    def fit_fold(tr_idx, val_idx, seed, fold):  # noqa: ANN001, ANN202
        enc = TargetEncoder(cv=5, smooth="auto", target_type="multiclass")
        te_tr = enc.fit_transform(te_tr_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_tr_view.iloc[val_idx])
        te_te = enc.transform(te_te_view)

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)],
                axis=1,
            )

        X_tr = assemble(X.iloc[tr_idx], te_tr)
        X_va = assemble(X.iloc[val_idx], te_va)
        X_te = assemble(X_test, te_te)
        model = HistGradientBoostingClassifier(**HGBC_CONFIG, random_state=seed)
        model.fit(X_tr, ds.y[tr_idx])
        assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
        print(f"    n_iter={model.n_iter_}")
        return model.predict_proba(X_va), model.predict_proba(X_te)

    oof, test, fold_scores = zoo_cv(
        ds, fit_fold, ckpt_name=f"hgbc_te3p_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3p", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3p_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
