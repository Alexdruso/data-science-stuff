"""ExtraTrees + RandomForest — bagging-tree diversity for the blend (PS S6E6).

The blend is GBDT-dominated and saturated at 0.9657; adding more boosters gets ~0
weight (correlated). Bagging trees are a genuinely different family (variance-
reduction vs boosting's bias-reduction) → potential decorrelation for the blend.
Honest expectation: each likely ~0.96 individually, value is in diversity, not
raw score. CV-gated; ensemble.py picks up oof_extratrees/oof_rf automatically.

sklearn trees need numeric input: the 2 categoricals are one-hot encoded (only 6
dummy columns). class_weight="balanced" mirrors the GBDTs.

Run:  python src/train_trees.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_FOLDS = 5

MODELS = {
    "extratrees": lambda: ExtraTreesClassifier(
        n_estimators=600, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42),
    "rf": lambda: RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42),
}


def main() -> None:
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))

    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    tr_pd, te_pd = train_pl.to_pandas(), test_pl.to_pandas()
    # one-hot the 2 cats, aligned across train/test
    combined = pd.concat([tr_pd[cat_cols], te_pd[cat_cols]], ignore_index=True)
    ohe = pd.get_dummies(combined, columns=cat_cols, dtype=np.float32)
    n = len(tr_pd)
    X = np.hstack([tr_pd[num_cols].to_numpy(np.float32), ohe.iloc[:n].to_numpy(np.float32)])
    X_test = np.hstack([te_pd[num_cols].to_numpy(np.float32), ohe.iloc[n:].to_numpy(np.float32)])
    test_ids = te_pd["id"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(tr_pd[TARGET].to_numpy())
    print(f"X {X.shape}  (num {len(num_cols)} + ohe {ohe.shape[1]})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for name, make in MODELS.items():
        oof = np.zeros((len(X), 3))
        test_proba = np.zeros((len(X_test), 3))
        fold_scores = []
        for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
            m = make()
            m.fit(X[tri], y[tri])
            oof[vai] = m.predict_proba(X[vai])
            test_proba += m.predict_proba(X_test) / N_FOLDS
            fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(oof[vai], axis=1))))
            print(f"  [{name}] fold {fold} balanced_acc: {fold_scores[-1]:.4f}")

        argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
        rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
        print(f"[{name}] OOF argmax: {argmax:.4f}  recall {dict(zip(le.classes_, rec.round(4)))}")
        tw, best = optimize_thresholds(oof, y)
        print(f"[{name}] threshold-tuned: {best:.4f}")
        save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{name}.json")
        save_cv_result(RESULTS_DIR, name, fold_scores, best, metric_name="balanced_acc")
        np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
        np.save(RESULTS_DIR / f"test_{name}.npy", test_proba)
        labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{name}.csv", index=False)
        print(f"[{name}] saved oof/test/submission\n")


if __name__ == "__main__":
    main()
