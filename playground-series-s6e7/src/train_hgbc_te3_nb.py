"""Original-world naive-Bayes prior columns (Day-10, GF #2).

The generator fingerprint showed the ORIGINAL 50k's features are essentially
independent given nothing (|corr| 0.003) — so naive Bayes is ~correctly
specified THERE, while the synthetic train (corr ~0.10, distorted value
frequencies) is what our TE tables see. Adds 3 log-posterior columns
log P_orig(y=c) + sum_col log P_orig(v_col | c) fit ENTIRELY on the external
dataset (label-free w.r.t. competition rows — no fold logic, no leak):
a clean pre-generator prior where the TE counts are distorted. Numerics are
decile-binned (train quantiles) for count stability; competition NaN bins
marginalize (term skipped — original has zero missing). Distinct from the
dead external-row augmentation (Day-3) and the dead synthetic-world analytic
posterior (Day-9).

Paired vs hgbc_te3_u_s42 (0.9504): same TE block, same tuned config, +3 cols.

Run (unrepaired namespace):
  S6E7_SEEDS=42 S6E7_RUN_TAG=_u_s42 python src/train_hgbc_te3_nb.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES, NUM_COLS, TARGET
from train_common import DATA_DIR, N_CLASSES, SEEDS, finalize, load_dataset
from train_hgbc_te3 import HGBC_CONFIG
from train_hgbc_te3_pairs import EXT_CSV, binned_view
from zoo_common import clear_ckpt, zoo_cv

warnings.filterwarnings("ignore", category=FutureWarning)

N_QUANT = 10
ALPHA = 1.0  # Laplace


def nb_log_posterior(
    view: pd.DataFrame, tables: dict, log_prior: np.ndarray
) -> np.ndarray:
    """(n, 3) NB log-posterior; missing/unseen values marginalize (skip term)."""
    out = np.tile(log_prior, (len(view), 1))
    for c, tab in tables.items():
        vals = view[c].to_numpy()
        known = pd.Series(vals).isin(tab.index).to_numpy()
        out[known] += tab.loc[vals[known]].to_numpy()
    out -= out.max(axis=1, keepdims=True)
    return out.astype(np.float32)


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ds = load_dataset()
    raw_cols = ds.feature_cols
    ext = pd.read_csv(EXT_CSV)
    y_ext = ext[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    num_cols = [c for c in NUM_COLS if c in raw_cols]
    edges = {
        c: np.unique(np.nanquantile(ds.train[c], np.linspace(0, 1, N_QUANT + 1)[1:-1]))
        for c in num_cols
    }
    bin_tr = binned_view(ds.train, raw_cols, edges)
    bin_te = binned_view(ds.test, raw_cols, edges)
    bin_ext = binned_view(ext, raw_cols, edges)

    # per-column log P_orig(v | c) with Laplace smoothing, fit on the ORIGINAL
    tables: dict[str, pd.DataFrame] = {}
    for c in raw_cols:
        ct = pd.crosstab(bin_ext[c], y_ext).reindex(columns=range(N_CLASSES), fill_value=0)
        tables[c] = np.log((ct + ALPHA) / (ct.sum(axis=0) + ALPHA * len(ct)))
    counts = np.bincount(y_ext, minlength=N_CLASSES).astype(float)
    log_prior = np.log(counts / counts.sum())

    nb_names = [f"nb_orig_{k}" for k in range(N_CLASSES)]
    nb_tr = nb_log_posterior(bin_tr, tables, log_prior)
    nb_te = nb_log_posterior(bin_te, tables, log_prior)
    print(f"NB prior columns from original: {nb_names}")

    X, X_test = ds.train.copy(), ds.test.copy()
    for col in ds.cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)
        cats = sorted(set(X[col].unique()) | set(X_test[col].unique()))
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    X[nb_names] = nb_tr
    X_test[nb_names] = nb_te
    te_tr_view = ds.train[raw_cols].astype(str)
    te_te_view = ds.test[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

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
        ds, fit_fold, ckpt_name=f"hgbc_te3nb_u_s{SEEDS[0]}", seed=SEEDS[0]
    )
    finalize("hgbc_te3nb", ds, oof, test, fold_scores)
    clear_ckpt(f"hgbc_te3nb_u_s{SEEDS[0]}")


if __name__ == "__main__":
    main()
