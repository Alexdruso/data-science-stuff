"""Port of cdeotte's RealMLP v1 (S6E6, notebook OOF 0.96881) to our stack.

Pure torch + sklearn (no RAPIDS). Uses realmlp_deotte.RealMLP_TD_Classifier (from-scratch
RealMLP with PBLD embeddings, n_ens=8, NTP linears, EMA, flat_cos, and balanced-softmax loss
loss_prior_power=1.075) + the notebook's modest feature_engineering + per-fold TargetEncoder on
the two interaction-combo cats. Train/test sorted by id so oof/test arrays match our id-sorted
convention → drop into build_lr_stack / build_gbdt_stack. This is a STRONG NON-GBDT base (the
diversity the stacker rewards most) — the one our every prior NN (~0.95) never reached.

GATE: standalone OOF ~0.9688; then add oof_realmlp_deotte/test_realmlp_deotte to the stack.

Run:  python src/train_realmlp_deotte.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from realmlp_deotte import CONFIG, RealMLP_TD_Classifier

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT_DIR = RESULTS_DIR / "_realmlp_deotte_ckpt"
ID, TARGET = "id", "class"
CLASS_MAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
SEED, FOLDS = 42, 5
RUN = "realmlp_deotte"

color_pairs = [("u", "g"), ("u", "r")]
important_combos = sorted([("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")])


def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    df["_g_/_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).astype("float32")
    df["_i_/_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).astype("float32")
    for a, b in color_pairs:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[col])}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[col] = codes
        df[col] = df[col].astype("category")
    for col in num_cols:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[col])}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")
    for col, bins_list in {"delta": [100, 500]}.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                category_map[bin_name] = kb
            else:
                binned = category_map[bin_name].transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype("category")
    combo_names = []
    for cols in important_combos:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")
    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID).reset_index(drop=True)
    train[TARGET] = train[TARGET].map(CLASS_MAP)
    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID], axis=1)
    test_id = test[ID].to_numpy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    category_map: dict = {}
    X, new_cat_cols, _, _ = feature_engineering(X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, new_num_cols, combo_names = feature_engineering(X_test, cat_cols, num_cols, category_map, fit=False)
    cat_cols = sorted(cat_cols + new_cat_cols)
    num_cols += new_num_cols
    X = X.reindex(sorted(X.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)
    print(f"X {X.shape}  cat_cols {len(cat_cols)}  num_cols {len(num_cols)}  device {CONFIG['device']}")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    n_classes = y.nunique()
    oof = np.zeros((len(X), n_classes), dtype="float32")
    test_proba = np.zeros((len(X_test), n_classes), dtype="float32")
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        fold_seed = SEED + fold * 100
        cfg = {**CONFIG, "random_state": fold_seed}
        X_tr, X_val, X_tst = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        y_tr, y_val = y.iloc[tri], y.iloc[vai]

        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
        tr_enc = enc.fit_transform(X_tr[combo_names], y_tr)
        val_enc = enc.transform(X_val[combo_names])
        tst_enc = enc.transform(X_tst[combo_names])
        te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(n_classes)]
        X_tr[te_names] = tr_enc.astype("float32")
        X_val[te_names] = val_enc.astype("float32")
        X_tst[te_names] = tst_enc.astype("float32")
        X_tr = X_tr.reindex(sorted(X_tr.columns), axis=1)
        X_val = X_val.reindex(sorted(X_val.columns), axis=1)
        X_tst = X_tst.reindex(sorted(X_tst.columns), axis=1)
        if fold == 1:
            print(f"  n_features {X_tr.shape[1]}  cat {len(cat_cols)}  TE {len(te_names)}")

        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        model = RealMLP_TD_Classifier(**cfg)
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols,
                  ckpt_path=str(CKPT_DIR / f"fold{fold}.pth"), X_test=X_tst)
        oof[vai] = model.best_val_probs_.astype("float32")
        test_proba += model.predict_proba(X_tst).astype("float32") / FOLDS
        fold_scores.append(float(balanced_accuracy_score(y_val, np.argmax(oof[vai], axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}")
        del model, X_tr, X_val, X_tst, y_tr, y_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_np = y.to_numpy()
    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y_np, pred))
    rec = recall_score(y_np, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [notebook 0.96881; xgb_deotte 0.96699]")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y_np)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, ["GALAXY", "QSO", "STAR"], RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INV_CLASS_MAP[i] for i in np.argmax(test_proba, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID: test_id, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
