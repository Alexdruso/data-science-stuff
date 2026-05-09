"""Post-ensemble submission blending for PS S6E5.

Two techniques (from public competition notebooks):
  1. Rank blend: blend ranks of anchor + support, re-assign anchor's sorted values.
     Preserves calibration; only changes ordering.
  2. Selective consensus correction: build consensus from support submissions, apply
     a small blend weight only on the top-k% most disagreed-upon rows.

Both are evaluated on OOF arrays (where ground truth exists), then the best
parameterisation is applied to test predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import build_features, compute_group_features

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TARGET = "PitNextLap"

ANCHOR_OOF_PATH = RESULTS_DIR / "oof_ensemble.npy"
ANCHOR_TEST_CSV = SUBMISSIONS_DIR / "ensemble_v4.csv"

SUPPORT_NAMES = ["lgbm", "lgbm_ar", "xgboost", "catboost", "mlp", "rnn"]


# ---------------------------------------------------------------------------
# Blending functions
# ---------------------------------------------------------------------------

def rank_blend(anchor: np.ndarray, support: np.ndarray, w: float) -> np.ndarray:
    """Blend ranks, then re-assign anchor's sorted probability values."""
    n = len(anchor)
    a_rank = np.argsort(np.argsort(anchor, kind="mergesort"), kind="mergesort") / n
    s_rank = np.argsort(np.argsort(support, kind="mergesort"), kind="mergesort") / n
    blended = (1.0 - w) * a_rank + w * s_rank
    order = np.argsort(blended, kind="mergesort")
    out = np.empty_like(anchor)
    out[order] = np.sort(anchor)
    return out


def selective_correction(
    anchor: np.ndarray,
    consensus: np.ndarray,
    top_k_pct: float,
    blend_w: float,
    variant: str,
) -> np.ndarray:
    """Apply blend_w * consensus correction only on the most disagreed-upon rows."""
    out = anchor.copy()
    delta = np.abs(consensus - anchor)

    if variant == "up":
        mask_direction = consensus > anchor
    elif variant == "down":
        mask_direction = consensus < anchor
    else:
        mask_direction = np.ones(len(anchor), dtype=bool)

    k = max(1, int(len(anchor) * top_k_pct / 100.0))
    # Among directional rows, pick the top-k by |delta|
    candidates = np.where(mask_direction)[0]
    if len(candidates) == 0:
        return out
    top_candidates = candidates[np.argsort(delta[candidates])[::-1][:k]]
    out[top_candidates] = (1.0 - blend_w) * anchor[top_candidates] + blend_w * consensus[top_candidates]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load labels for OOF evaluation ---
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    train = compute_group_features(train_raw, build_features(train_raw))
    y = train[TARGET].to_numpy()

    # --- Load anchor ---
    anchor_oof = np.load(ANCHOR_OOF_PATH)
    anchor_test_df = pd.read_csv(ANCHOR_TEST_CSV)
    anchor_test = anchor_test_df[TARGET].to_numpy()
    test_ids = anchor_test_df["id"].to_numpy()
    print(f"Anchor OOF AUC: {roc_auc_score(y, anchor_oof):.4f}")

    # --- Load support predictions ---
    support_oofs: dict[str, np.ndarray] = {}
    support_tests: dict[str, np.ndarray] = {}
    for name in SUPPORT_NAMES:
        oof_path = RESULTS_DIR / f"oof_{name}.npy"
        test_path = RESULTS_DIR / f"test_{name}.npy"
        if oof_path.exists() and test_path.exists():
            support_oofs[name] = np.load(oof_path)
            support_tests[name] = np.load(test_path)

    # --- Step 1: Diversity analysis ---
    print("\n--- Diversity analysis (vs anchor test predictions) ---")
    print(f"  {'Model':<12} {'corr':>7} {'mean_abs_delta':>15} {'oof_auc':>8}")
    all_high_corr = True
    for name, oof in support_oofs.items():
        test_arr = support_tests[name]
        corr = float(pearsonr(anchor_test, test_arr)[0])
        mad = float(np.mean(np.abs(test_arr - anchor_test)))
        oof_auc = float(roc_auc_score(y, oof))
        flag = " ← diverse" if corr < 0.999 or mad > 0.001 else ""
        print(f"  {name:<12} {corr:>7.4f} {mad:>15.5f} {oof_auc:>8.4f}{flag}")
        if corr < 0.999 or mad > 0.001:
            all_high_corr = False
    if all_high_corr:
        print("  ⚠ All support submissions corr > 0.999 and delta < 0.001 — signal likely exhausted.")

    # --- Step 2: Rank blend grid search on OOF ---
    print("\n--- Rank blend OOF grid (anchor OOF AUC = {:.4f}) ---".format(roc_auc_score(y, anchor_oof)))
    best_rank = {"auc": roc_auc_score(y, anchor_oof), "name": None, "w": None, "oof": anchor_oof, "test": anchor_test}
    for name, oof in support_oofs.items():
        for w in [0.01, 0.02, 0.05, 0.10]:
            blended_oof = rank_blend(anchor_oof, oof, w)
            auc = float(roc_auc_score(y, blended_oof))
            marker = " ✓ NEW BEST" if auc > best_rank["auc"] else ""
            print(f"  rank_blend({name}, w={w:.2f}): OOF AUC {auc:.4f}{marker}")
            if auc > best_rank["auc"]:
                blended_test = rank_blend(anchor_test, support_tests[name], w)
                best_rank = {"auc": auc, "name": name, "w": w, "oof": blended_oof, "test": blended_test}

    # --- Step 3: Selective consensus correction grid search on OOF ---
    print("\n--- Selective consensus correction OOF grid ---")
    floor = 0.93
    raw_weights = {n: max(0.0, float(roc_auc_score(y, o)) - floor) for n, o in support_oofs.items()}
    total_w = sum(raw_weights.values())
    consensus_oof = sum(
        (raw_weights[n] / total_w) * o for n, o in support_oofs.items()
    )
    consensus_test = sum(
        (raw_weights[n] / total_w) * support_tests[n] for n in support_oofs
    )
    print(f"  Consensus OOF AUC: {roc_auc_score(y, consensus_oof):.4f}")
    print(f"  Weights: " + ", ".join(f"{n}={raw_weights[n]/total_w:.3f}" for n in support_oofs))

    best_sel = {"auc": roc_auc_score(y, anchor_oof), "params": None, "oof": anchor_oof, "test": anchor_test}
    for top_k in [1, 2, 5]:
        for blend_w in [0.05, 0.10, 0.20]:
            for variant in ["all", "up", "down"]:
                corrected_oof = selective_correction(anchor_oof, consensus_oof, top_k, blend_w, variant)
                auc = float(roc_auc_score(y, corrected_oof))
                marker = " ✓ NEW BEST" if auc > best_sel["auc"] else ""
                print(f"  selective(k={top_k}%, w={blend_w:.2f}, {variant}): OOF AUC {auc:.4f}{marker}")
                if auc > best_sel["auc"]:
                    corrected_test = selective_correction(anchor_test, consensus_test, top_k, blend_w, variant)
                    best_sel = {
                        "auc": auc,
                        "params": f"k={top_k}% blend_w={blend_w} variant={variant}",
                        "oof": corrected_oof,
                        "test": corrected_test,
                    }

    # --- Step 4: Pick overall best and save if it beats anchor ---
    anchor_auc = float(roc_auc_score(y, anchor_oof))
    candidates = [
        ("rank_blend", best_rank["auc"], best_rank["oof"], best_rank["test"],
         f"support={best_rank['name']} w={best_rank['w']}"),
        ("selective", best_sel["auc"], best_sel["oof"], best_sel["test"],
         best_sel["params"]),
    ]
    strategy, best_auc, best_oof, best_test, params = max(candidates, key=lambda t: t[1])

    print(f"\nBest overall: {strategy} ({params})  OOF AUC: {best_auc:.4f}")

    # Require a meaningful improvement (≥ 0.0001) to avoid saving floating-point noise
    if best_auc <= anchor_auc + 1e-4:
        print(f"Blend signal exhausted — best ({best_auc:.6f}) does not meaningfully beat anchor ({anchor_auc:.6f}). No submission saved.")
        return

    print(f"New best {best_auc:.4f} > anchor {anchor_auc:.4f}. Saving blend_v1.")

    save_cv_result(RESULTS_DIR, "blend_v1", [], best_auc)
    np.save(RESULTS_DIR / "oof_blend_v1.npy", best_oof)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, TARGET: best_test})
    out_path = SUBMISSIONS_DIR / "blend_v1.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


if __name__ == "__main__":
    main()
