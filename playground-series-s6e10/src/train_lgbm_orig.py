"""PR-1: original-data augmentation, paired against its own control.

    python src/train_lgbm_orig.py

Runs two arms on identical folds, params and device (CPU):
  base — synthetic training rows only (`is_original` constant 0)
  aug  — each fold's training split + all original rows (`is_original` = 1)
Original rows are appended to the TRAINING split only; validation and test are
pure synthetic, so OOF keeps measuring the distribution the LB is scored on.
The bar is pre-registered in CLAUDE.md (PR-1) — do not move it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from baseline import LGBM_PARAMS
from cv_results import save_cv_result
from matrices import Matrices, build_matrices

from data_science_stuff.kaggle.cv import run_cv
from data_science_stuff.kaggle.io import competition_dirs

_, RESULTS_DIR, _ = competition_dirs(__file__)

# CPU for both arms so the pair is clean; 8 threads leaves cores for the GPU legs.
PARAMS: dict[str, object] = {**LGBM_PARAMS, "device_type": "cpu", "n_jobs": 8}


def run_arm(
    m: Matrices, name: str, extra: tuple[pd.DataFrame, np.ndarray] | None
) -> list[float]:

    def fit_fold(
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_va: pd.DataFrame,
        y_va: np.ndarray,
        x_te: pd.DataFrame,
        _fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if extra is not None:
            x_tr = pd.concat([x_tr, extra[0]], ignore_index=True)
            y_tr = np.concatenate([y_tr, extra[1]])
        model = LGBMClassifier(**PARAMS)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        print(f"    [{name}] best_iter={model.best_iteration_}", flush=True)
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]

    def fold_score(y_va: np.ndarray, val_pred: np.ndarray) -> float:
        return float(roc_auc_score(y_va, val_pred))

    oof, test, fold_scores = run_cv(
        m.X, m.y, m.X_test, fit_fold, n_outputs=1, score_fn=fold_score
    )
    oof_auc = float(roc_auc_score(m.y, oof))
    print(f"  [{name}] folds {[round(s, 5) for s in fold_scores]}  OOF {oof_auc:.5f}")
    save_cv_result(RESULTS_DIR, name, fold_scores, oof_auc, metric_name="auc")
    np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
    np.save(RESULTS_DIR / f"test_{name}.npy", test)
    return [*fold_scores, oof_auc]


def main() -> None:
    m = build_matrices(with_original=True)
    assert m.X_orig is not None and m.y_orig is not None
    print(f"train {m.X.shape}  original {m.X_orig.shape}  test {m.X_test.shape}")
    base = run_arm(m, "lgbm_pr1_base", None)
    aug = run_arm(m, "lgbm_pr1_aug", (m.X_orig, m.y_orig))
    deltas = np.array(aug) - np.array(base)
    n_pos = int((deltas[:5] > 0).sum())
    verdict = "PASS" if deltas[5] >= 0.0005 and n_pos >= 4 else "FAIL"
    print(f"\nPR-1 per-fold delta {np.round(deltas[:5], 5).tolist()}")
    print(f"PR-1 OOF delta {deltas[5]:+.5f}  positive folds {n_pos}/5  → {verdict}")


if __name__ == "__main__":
    main()
