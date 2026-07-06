"""Region-wise (mixture-of-experts) blending for PS S6E7.

The error structure is bimodal: rows with all 3 key drivers present are near the noise
floor (bacc ~0.972) while rows with >=1 missing driver collapse (bacc ~0.886). Models may
be differentially good per regime (the MLP sees explicit missingness features; CatBoost
CTRs handle missing categories differently) — so fit Nelder-Mead blend weights AND
per-class decision weights SEPARATELY per regime, on OOF only. Pure post-processing of
existing arrays; no retraining.

GATE (per lb_anchor finding): offline metrics cannot rank <0.001 differences. Queue this
for a tomorrow submission ONLY if it beats the same-arrays global blend by >0.001 on plain
OOF bacc AND directionally on adv-weighted bacc; otherwise record as flat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict
from features import CLASSES, TARGET
from lb_anchor import KEY_DRIVERS, correct, weighted_bacc
from train_common import DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR, load_dataset

MODELS = ["lgbm", "xgboost", "catboost", "mlp"]
SUFFIX = "_bag"  # arrays to operate on (bagged set incl. MLP)


def fit_blend(oofs: list[NDArray], y: NDArray, idx: NDArray) -> NDArray:
    """NM softmax blend weights fit on the given OOF row subset."""

    def blend(w: NDArray) -> NDArray:
        s = np.exp(w) / np.exp(w).sum()
        return sum(s[i] * a[idx] for i, a in enumerate(oofs))

    res = minimize(
        lambda w: -float(balanced_accuracy_score(y[idx], np.argmax(blend(w), axis=1))),
        np.ones(len(oofs)),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    return np.exp(res.x) / np.exp(res.x).sum()


def apply_blend(w: NDArray, arrays: list[NDArray], idx: NDArray) -> NDArray:
    return sum(w[i] * a[idx] for i, a in enumerate(arrays))


def miss_mask(csv: str) -> NDArray:
    return (
        pl.read_csv(DATA_DIR / csv)
        .sort("id")
        .select(pl.any_horizontal([pl.col(c).is_null() for c in KEY_DRIVERS]))
        .to_numpy()
        .ravel()
    )


def main() -> None:
    ds = load_dataset()
    y = ds.y
    oof = {m: np.load(RESULTS_DIR / f"oof_{m}{SUFFIX}.npy") for m in MODELS}
    test = {m: np.load(RESULTS_DIR / f"test_{m}{SUFFIX}.npy") for m in MODELS}
    adv = np.load(RESULTS_DIR / "adv_scores_train.npy")

    # Precorrect each model by its own decision weights (deployed surface), as ensemble.py does.
    w_model = {m: tune_decision_weights(y, oof[m]) for m in MODELS}
    pc_oof = [correct(oof[m], w_model[m]) for m in MODELS]
    pc_test = [correct(test[m], w_model[m]) for m in MODELS]

    m_tr = miss_mask("train.csv")
    m_te = miss_mask("test.csv")
    print(
        f"missing-driver rows: train {100 * m_tr.mean():.1f}%  test {100 * m_te.mean():.1f}%"
    )

    # ── global (control, same arrays) ────────────────────────────────────────
    all_idx = np.arange(len(y))
    gw = fit_blend(pc_oof, y, all_idx)
    g_oof = apply_blend(gw, pc_oof, all_idx)
    g_dw = tune_decision_weights(y, g_oof)
    g_pred = weighted_predict(g_oof, g_dw)
    g_plain = balanced_accuracy_score(y, g_pred)
    g_adv = weighted_bacc(y, g_pred, adv)
    print(
        f"\nGLOBAL   blend {dict(zip(MODELS, gw.round(3)))}  plain={g_plain:.5f}  advwt={g_adv:.5f}"
    )

    # ── region-wise: separate blend + decision weights per regime ────────────
    pred = np.zeros(len(y), dtype=np.int64)
    test_pred = np.zeros(len(m_te), dtype=np.int64)
    for name, tr_mask, te_mask in [
        ("complete", ~m_tr, ~m_te),
        ("missing", m_tr, m_te),
    ]:
        idx = np.where(tr_mask)[0]
        rw = fit_blend(pc_oof, y, idx)
        r_oof = apply_blend(rw, pc_oof, idx)
        r_dw = tune_decision_weights(y[idx], r_oof)
        pred[idx] = weighted_predict(r_oof, r_dw)
        te_idx = np.where(te_mask)[0]
        r_test = apply_blend(rw, pc_test, te_idx)
        test_pred[te_idx] = weighted_predict(r_test, r_dw)
        sub_b = balanced_accuracy_score(y[idx], pred[idx])
        print(
            f"{name:>8} blend {dict(zip(MODELS, rw.round(3)))}  subset_bacc={sub_b:.5f}"
        )

    r_plain = balanced_accuracy_score(y, pred)
    r_adv = weighted_bacc(y, pred, adv)
    print(f"\nREGIONAL  plain={r_plain:.5f}  advwt={r_adv:.5f}")
    print(
        f"delta vs global: plain {r_plain - g_plain:+.5f}  advwt {r_adv - g_adv:+.5f}"
    )

    if (r_plain - g_plain) > 0.001 and (r_adv - g_adv) > 0:
        labels = np.array(CLASSES)[test_pred]
        out = SUBMISSIONS_DIR / "region_blend.csv"
        pl.DataFrame({"id": ds.test_ids, TARGET: labels}).write_csv(out)
        print(f"LARGE margin → queued for tomorrow: {out}")
    else:
        print(
            "VERDICT: below the >0.001 gate → FLAT, not queued (recorded in CLAUDE.md)."
        )


if __name__ == "__main__":
    main()
