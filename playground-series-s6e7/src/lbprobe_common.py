"""LB-probe shared plumbing (quarantined curiosity track — see lbprobe_ingest.py).

Loads the reconciled score table, stacks submission CSVs into an aligned int8
label matrix, computes prediction-profile groups, and builds a calibrated
per-row class-probability prior from a stack's test/OOF arrays.
"""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from features import CLASSES, TARGET, build_features, label_to_idx

COMP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = COMP_DIR / "data"
RESULTS_DIR = COMP_DIR / "results"
SUBS_DIR = COMP_DIR / "submissions"

M_TEST = 295_753
P_PUBLIC_DEFAULT = 0.2
ROUND_SD = 1e-5 / np.sqrt(12.0)  # 5-dp display rounding, uniform on the grid cell


def load_test_ids() -> np.ndarray:
    return build_features(pl.read_csv(DATA_DIR / "test.csv"))["id"].to_numpy()


def load_train_y() -> np.ndarray:
    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    return np.array([label_to_idx[v] for v in train[TARGET].to_list()], dtype=np.int8)


def load_scores() -> pl.DataFrame:
    """One row per usable file: (file_name, public_score), best-documented dedupe."""
    df = pl.read_csv(RESULTS_DIR / "lbprobe_scores.csv")
    return (
        df.filter(pl.col("reliable"))
        .group_by("file_name")
        .agg(pl.col("public_score").first())
        .sort("file_name")
    )


_SUB_CACHE: dict[str, np.ndarray] = {}


def load_submission_matrix(
    file_names: list[str], test_ids: np.ndarray
) -> np.ndarray:
    """Stack hard-label submissions into an (M, K) int8 matrix in sorted-id order."""
    cols = []
    for fname in file_names:
        if fname not in _SUB_CACHE:
            sub = pl.read_csv(SUBS_DIR / fname).sort("id")
            assert sub.height == M_TEST, f"{fname}: {sub.height} rows"
            assert np.array_equal(
                sub["id"].to_numpy(), test_ids
            ), f"{fname}: id mismatch"
            _SUB_CACHE[fname] = (
                sub[TARGET].replace_strict(label_to_idx, return_dtype=pl.Int8).to_numpy()
            )
        cols.append(_SUB_CACHE[fname])
    return np.stack(cols, axis=1)


def profile_groups(y_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group rows by identical prediction profiles.

    Returns (group_id per row int32, unique profile matrix (G, K) int8).
    """
    y_c = np.ascontiguousarray(y_mat)
    void = y_c.view(np.dtype((np.void, y_c.dtype.itemsize * y_c.shape[1])))
    _, first_idx, inv = np.unique(void.ravel(), return_index=True, return_inverse=True)
    return inv.astype(np.int32), y_c[first_idx]


_PRIOR_CACHE: dict[str, np.ndarray] = {}


def calibrated_prior(stack_name: str = "lrstack_u3_r", quiet: bool = False) -> np.ndarray:
    if stack_name in _PRIOR_CACHE:
        return _PRIOR_CACHE[stack_name]
    q = _calibrated_prior_impl(stack_name, quiet)
    _PRIOR_CACHE[stack_name] = q
    return q


def _calibrated_prior_impl(stack_name: str, quiet: bool) -> np.ndarray:
    """Test-row class posterior from a stack, bias-recalibrated on its honest OOF.

    The LR meta was fit class-balanced, so raw probabilities are tilted toward
    uniform priors. Fit an additive per-class logit bias by OOF NLL and apply
    it to the test array; the result should average close to the train priors.
    """
    oof = np.load(RESULTS_DIR / f"oof_{stack_name}.npy").astype(np.float64)
    test = np.load(RESULTS_DIR / f"test_{stack_name}.npy").astype(np.float64)
    y = load_train_y()
    log_oof = np.log(np.clip(oof, 1e-12, None))

    def nll(b2: np.ndarray) -> float:
        b = np.array([0.0, b2[0], b2[1]])
        z = log_oof + b
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    res = minimize(nll, x0=np.zeros(2), method="Nelder-Mead")
    b = np.array([0.0, res.x[0], res.x[1]])
    z = np.log(np.clip(test, 1e-12, None)) + b
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    if not quiet:
        print(
            f"prior calibration [{stack_name}]: bias={b.round(4)}, "
            f"raw test priors={test.mean(axis=0).round(4)}, "
            f"calibrated={q.mean(axis=0).round(4)} "
            f"(train priors: 0.8590 / 0.0577 / 0.0833)"
        )
    return q


def decode_labels(pred_idx: np.ndarray) -> np.ndarray:
    return np.array(CLASSES)[pred_idx]
