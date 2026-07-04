"""Low-z STAR-vs-GALAXY boundary specialist — mixture-of-experts probe (S6E6).

The incumbent error is almost entirely STAR<->GALAXY confusion at low redshift (z<~0.2,
~75k rows, the model "77% confident in the wrong class"). Every generalist spreads capacity
across all redshifts; this trains a binary STAR-vs-GALAXY model ONLY on the low-z zone, so
its whole capacity goes to the hard boundary. Two questions:

  1. DIAGNOSTIC (reducible vs irreducible): does focusing on the region beat a GENERALIST
     control (same model/features/config, trained on ALL rows) on the SAME held-out low-z
     STAR/GALAXY rows? If yes -> the overlap is reducible with focused capacity; if it ties
     -> we've PROVEN the low-z overlap is irreducible for these features (not just assumed it).
  2. STACK VALUE: feed the specialist's P(STAR) as a new column into the 7-base LR stack and
     read the argmax delta — a column the generalist bases don't supply.

Honest CV: outer 5-fold over ALL train rows (rs=42). In each fold the specialist trains on the
low-z STAR+GALAXY rows of the TRAIN split only and predicts the held-out fold -> clean OOF.

Run:  python src/train_lowz_specialist.py [--thresh 0.25]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from lgbm_device import get_lgbm_device

from data_science_stuff.kaggle.io import competition_dirs

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
STACK_MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
                "xgb_deotte", "realmlp_deotte", "catboost_deotte"]
NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
            "u_g", "g_r", "r_i", "i_z", "u_z", "log1p_redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
EPS = 1e-6
# GALAXY=0, QSO=1, STAR=2 after LabelEncoder (alphabetical)
GAL, STAR = 0, 2


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS))


def lgb_params(seed, dev_type, n_jobs):
    return dict(
        objective="binary", n_estimators=600, learning_rate=0.03, num_leaves=63,
        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, class_weight="balanced", random_state=seed, verbose=-1,
        n_jobs=n_jobs, device_type=dev_type,
    )


def to_frame(df: pl.DataFrame):
    """LGBM-ready pandas frame: numerics + category-dtype cats."""
    pdf = df.select(NUM_COLS + CAT_COLS).to_pandas()
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype("category")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.25)
    args = ap.parse_args()
    dev_type, n_jobs = get_lgbm_device()

    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())  # 0=GAL 1=QSO 2=STAR
    Xtr_all, Xte_all = to_frame(tr), to_frame(te)
    z = tr["redshift"].to_numpy()

    lowz = z < args.thresh
    sg = (y == GAL) | (y == STAR)  # star-vs-galaxy rows
    print(f"thresh z<{args.thresh}: low-z rows {lowz.sum()} "
          f"({(y[lowz] == STAR).sum()} STAR / {(y[lowz] == GAL).sum()} GAL / "
          f"{(y[lowz] == 1).sum()} QSO); low-z STAR+GAL = {(lowz & sg).sum()}")

    # binary label STAR=1 else 0 (only meaningful on STAR/GAL rows)
    ybin = (y == STAR).astype(int)

    # full-length OOF specialist P(STAR) column (clean: trained on TRAIN split low-z SG only)
    spec_oof = np.zeros(len(y))
    # diagnostic accumulators on held-out low-z STAR/GAL rows
    diag_idx, diag_spec, diag_ctrl = [], [], []

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tri, vai in skf.split(Xtr_all, y):
        tr_sg = tri[sg[tri]]                 # all STAR/GAL in train split
        tr_spec = tri[lowz[tri] & sg[tri]]   # low-z STAR/GAL in train split

        spec = LGBMClassifier(**lgb_params(42, dev_type, n_jobs))
        spec.fit(Xtr_all.iloc[tr_spec], ybin[tr_spec], categorical_feature=CAT_COLS)
        ctrl = LGBMClassifier(**lgb_params(42, dev_type, n_jobs))
        ctrl.fit(Xtr_all.iloc[tr_sg], ybin[tr_sg], categorical_feature=CAT_COLS)

        # specialist column for ALL held-out rows (extrapolates outside low-z; stacker decides)
        spec_oof[vai] = spec.predict_proba(Xtr_all.iloc[vai])[:, 1]

        # diagnostic: held-out low-z STAR/GAL rows only
        va_sg = vai[lowz[vai] & sg[vai]]
        diag_idx.append(va_sg)
        diag_spec.append(spec.predict_proba(Xtr_all.iloc[va_sg])[:, 1])
        diag_ctrl.append(ctrl.predict_proba(Xtr_all.iloc[va_sg])[:, 1])

    di = np.concatenate(diag_idx)
    yb = ybin[di]
    spec_ba = balanced_accuracy_score(yb, (np.concatenate(diag_spec) > 0.5).astype(int))
    ctrl_ba = balanced_accuracy_score(yb, (np.concatenate(diag_ctrl) > 0.5).astype(int))
    print(f"\nDIAGNOSTIC on held-out low-z STAR/GAL ({len(di)} rows):")
    print(f"  specialist (low-z only)  binary balanced-acc {spec_ba:.4f}")
    print(f"  generalist control (all) binary balanced-acc {ctrl_ba:.4f}")
    print(f"  focus delta {spec_ba - ctrl_ba:+.4f}  "
          f"({'REDUCIBLE — focus helps' if spec_ba - ctrl_ba > 0.002 else 'irreducible — focus ties/loses'})")

    # test column: specialist fit on ALL low-z STAR+GAL train rows
    fit_idx = np.where(lowz & sg)[0]
    spec_full = LGBMClassifier(**lgb_params(42, dev_type, n_jobs))
    spec_full.fit(Xtr_all.iloc[fit_idx], ybin[fit_idx], categorical_feature=CAT_COLS)
    spec_test = spec_full.predict_proba(Xte_all)[:, 1]
    np.save(RESULTS_DIR / "oof_lowz_spec.npy", spec_oof)
    np.save(RESULTS_DIR / "test_lowz_spec.npy", spec_test)

    # STACK VALUE: 7-base LR stack with vs without the specialist column
    base_tr = [logit(np.load(RESULTS_DIR / f"oof_{m}.npy")) for m in STACK_MODELS]
    spec_block_tr = np.column_stack([logit(spec_oof), logit(1 - spec_oof)])

    def run(blocks):
        Z = np.hstack(blocks)
        oof = np.zeros((len(y), 3))
        for seed in [2024, 7, 13]:
            skf2 = StratifiedKFold(5, shuffle=True, random_state=seed)
            for tri, vai in skf2.split(Z, y):
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
                lr.fit(Z[tri], y[tri])
                oof[vai] += lr.predict_proba(Z[vai]) / 3
        return float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))

    base = run(base_tr)
    withspec = run(base_tr + [spec_block_tr])
    print(f"\nSTACK 7-base {base:.5f} -> +specialist {withspec:.5f}  "
          f"(delta {withspec - base:+.5f})  [GBDT-stack incumbent 0.96981]")
    save_cv_result(RESULTS_DIR, "lowz_spec", [], withspec, metric_name="balanced_acc")


if __name__ == "__main__":
    main()
