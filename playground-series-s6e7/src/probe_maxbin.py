"""Day-23 (post-deadline post-mortem): does `max_bin` matter? (fryhat's discussion post)

WHY. A competitor at essentially our public score (0.95049 vs our 0.95036 - a 0.09
public-sd gap) posted their experiment log, and the one knob in it this campaign never
swept is **max_bin=768**. Our LGBM/XGB te3 configs hardcode `max_bin: 237` (inherited
verbatim from Susol's HGBC config on Day-9), and `tune_hgbc_te3.py` only ever searched
max_bins in [192, 255] -- sklearn's HistGradientBoosting caps at 255, so the axis was
structurally invisible to our tree canary and we never re-opened it on a library that
allows more.

MECHANISM THAT MAKES IT PLAUSIBLE (our own EDA, Day-4): the numerics are COARSE QUANTIZED
GRIDS -- water_intake 400 uniques, heart_rate exactly 1dp, calorie/steps integer. With
max_bin=237 the histogram builder merges distinct grid values into shared bins, so splits
cannot land on exact values. A high max_bin lets them.

COUNTER-ARGUMENT (why it may be nothing here): our te13 block already target-encodes all
13 columns at EXACT VALUE resolution, so the exact-value information is arguably already
in the matrix by another route -- and `hgbc_te3tri` (the exact 3-way rule-triple TE) read
FLAT. If te13 has already spanned it, max_bin should do nothing on this recipe. That is
the actual question: is max_bin a second vehicle for information te13 already carries?

DESIGN: paired 5-fold, seed 42, identical folds and identical seeded TE inner-CV across
arms -- only `max_bin` varies. Reports the deployed metric (decision-weighted BA) and
plain argmax. Zero risk to the finals: writes nothing under any finals-lineage key.

NOTE ON STATUS: the finals are already selected and this cannot change them. This runs to
answer "did we leave something real on the table", for the record.

Run: python src/probe_maxbin.py -> results/probe_maxbin.txt
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from fast_dw import fit_dw
from lgbm_device import get_lgbm_device
from train_common import N_CLASSES, N_FOLDS, RESULTS_DIR, load_dataset
from train_lgbm_te3 import LGBM_TE3_PARAMS

from data_science_stuff.kaggle_utils import weighted_predict

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
BINS = [237, 512, 768, 1023]  # 237 = incumbent; 1023 = LightGBM's practical ceiling
OUT = RESULTS_DIR / "probe_maxbin.txt"

# ⚠️ CPU IS FORCED, and the reason is itself a finding. `get_lgbm_device()` returns
# ("cuda", 1) on this box (lightgbm 4.6.0 built with -DUSE_CUDA=1), and the CUDA
# histogram builder CANNOT handle max_bin > 255 -- the first run of this probe died with
# "[CUDA] an illegal memory access was encountered ... cuda_tree.cu 454". So there were
# TWO independent structural reasons this axis was never explored in 23 days: sklearn's
# HGBC caps max_bins at 255 (our tuning canary), and the CUDA LGBM build crashes above it.
# Neither is a decision anyone made; both are silent constraints of the tooling.
DEVICE = {"device_type": "cpu", "n_jobs": -1}
# 2-fold (folds 1 and 3) cheap objective -- the same design tune_hgbc_te3.py used for its
# 311-trial search. Stated, not hidden: this has less power than a 5-fold read, so it is a
# SCREEN. A |delta| near the +0.0007 screen bar would warrant the full 5-fold confirmation.
FOLD_SUBSET = [1, 3]


def ba(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, pred))


def main() -> None:
    ds = load_dataset()
    raw_cols = ds.feature_cols
    X = ds.train.copy()
    for col in ds.cat_cols:
        X[col] = X[col].astype("category")
    te_view = ds.train[raw_cols].astype(str)
    te_names = [f"te_{c}_{k}" for c in raw_cols for k in range(N_CLASSES)]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [(i, f) for i, f in enumerate(skf.split(ds.train, ds.y), 1) if i in FOLD_SUBSET]
    oofs = {b: np.zeros((len(ds.y), N_CLASSES)) for b in BINS}
    scored = np.concatenate([v for _, (_, v) in folds])  # only these rows are populated

    for fold, (tr_idx, val_idx) in folds:
        t0 = time.time()
        # identical TE block for every arm in this fold => the only variable is max_bin
        enc = TargetEncoder(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            smooth="auto",
            target_type="multiclass",
        )
        te_tr = enc.fit_transform(te_view.iloc[tr_idx], ds.y[tr_idx])
        te_va = enc.transform(te_view.iloc[val_idx])

        def assemble(base: pd.DataFrame, blk: np.ndarray) -> pd.DataFrame:
            return pd.concat(
                [base.reset_index(drop=True), pd.DataFrame(blk, columns=te_names)], axis=1
            )

        X_tr, X_va = assemble(X.iloc[tr_idx], te_tr), assemble(X.iloc[val_idx], te_va)
        for b in BINS:
            m = LGBMClassifier(
                **{**LGBM_TE3_PARAMS, **DEVICE, "max_bin": b, "random_state": SEED}
            )
            m.fit(
                X_tr,
                ds.y[tr_idx],
                eval_set=[(X_va, ds.y[val_idx])],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            assert list(m.classes_) == list(range(N_CLASSES))
            oofs[b][val_idx] = m.predict_proba(X_va)
        print(f"  fold {fold} done in {time.time() - t0:.0f}s", flush=True)

    lines = [
        "=" * 78,
        "probe_maxbin -- does max_bin matter on the te13 recipe? (LGBM, paired 5-fold, seed 42)",
        "  incumbent max_bin=237 (inherited from Susol Day-9); fryhats post used 768.",
        f"  2-fold SCREEN (folds {FOLD_SUBSET}) on {len(scored)} rows, LGBM forced to CPU",
        "  identical folds + identical seeded TE inner-CV across arms; only max_bin varies.",
        "=" * 78,
        f"  {'max_bin':>8}  {'dw-BA':>9}  {'argmax':>9}  {'d(dw)':>10}  {'d(argmax)':>10}",
    ]
    y_s = ds.y[scored]
    base_dw = base_am = None
    dws = {}
    for b in BINS:
        p = oofs[b][scored]
        dw = ba(y_s, weighted_predict(p, fit_dw(y_s, p)))
        am = ba(y_s, p.argmax(1))
        dws[b] = dw
        if base_dw is None:
            base_dw, base_am = dw, am
        lines.append(
            f"  {b:>8}  {dw:.6f}  {am:.6f}  {dw - base_dw:+.6f}  {am - base_am:+.6f}"
        )
    best = max(BINS, key=lambda b: dws[b])
    delta = dws[best] - base_dw
    lines += [
        "",
        f"  best arm max_bin={best}, delta vs incumbent {delta:+.6f}",
        "  READ: te13 already target-encodes all 13 cols at EXACT value resolution, so if",
        "  max_bin is a second vehicle for the same information this should be ~0. A large",
        "  positive would mean the campaign left a real axis unswept (sklearn HGBC caps at",
        "  255, which is why our tree canary could never see it).",
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    OUT.write_text(report + "\n")


if __name__ == "__main__":
    main()
