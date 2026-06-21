"""Class-conditional normalizing flows (S6E6 — the one untried MODEL FAMILY).

Every deotte-FE base (GBDT + RealMLP/TabM variants) is flat at the stack because they share an
error geometry — discriminative boundaries on the same features. This is orthogonal BY
CONSTRUCTION: fit a separate density p(x|class) with a normalizing flow, classify by Bayes
(argmax_c log p(x|c) + log prior_c). A generative model errs where the relative class DENSITY
is ambiguous, not where a discriminative margin is small → a genuinely different error set.

Design (locked with advisor):
  • Input = 6-dim BIJECTIVE basis [r, u_g, g_r, r_i, i_z, log1p_redshift]. This is invertible
    with (u,g,r,i,z,redshift): full continuous photometric+redshift info, NO degeneracy. Feeding
    bands AND exact-linear colors would put data on a lower-dim subspace → singular density →
    diverging log-probs. (spectral_type/galaxy_population are deterministic color cuts → already
    implied; alpha/delta are class-independent noise → excluded.)
  • THREE SEPARATE NSF flows (neural spline, not affine — affine ⇒ ~Gaussian ⇒ QDA, already
    flat). NOT one class-conditional flow: a shared backbone blurs class structure toward
    Gaussian. Independent flows can't blur; the per-class log-density offset is absorbed by the
    LR stacker's per-class intercept, so the scale-comparability "problem" is a non-issue.
  • Standardize with TRAIN-FOLD global mean/std (leak-safe). Early-stop each flow on its OWN
    class val NLL. Output softmax(log p(x|c)+log prior_c) → logit feeds the stacker.

Speed: fp32 (NOT fp16 — log-det/spline underflow → NaN; the one place AMP is wrong), fused
AdamW, 16k batches, all data GPU-resident. torch.compile DROPPED — measured net-negative on
zuko (it builds a fresh distribution object per call → Dynamo can't trace it; ~100x slower).

GATE: standalone ~0.95–0.96 is EXPECTED and acceptable (decorrelation is the deliverable, not
argmax). eval_stack_delta is binding. If flat too, that's a real finding (photometry-only
diversity exhausted), not an execution miss.

Run:  python src/train_flow.py [--save-logdens]

  --save-logdens  also save oof_flow_logdens.npy / test_flow_logdens.npy containing raw
                  log p(x|class) values (3 per row, NOT prior-adjusted, NOT softmaxed).
                  These are inputs for the density-augmented discriminative base: feed them
                  as 3 extra features into XGB/CatBoost on top of the deotte 240-FE.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import zuko
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from postprocess import optimize_thresholds, save_threshold_weights

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
ID = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]            # sorted → matches LabelEncoder used by the stacker
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
FLOW_COLS = ["r", "u_g", "g_r", "r_i", "i_z", "log1p_redshift"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, FOLDS = 42, 5
RUN = "flow"

# NSF / training hyperparams
N_TRANSFORMS, N_BINS, HIDDEN = 4, 8, (128, 128)
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 16384, 80, 6
EVAL_BS = 65536


def to_matrix(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack([df[c].to_numpy().astype(np.float32) for c in FLOW_COLS])


def train_class_flow(Xtr_c: np.ndarray, Xva_c: np.ndarray, seed: int):
    """One NSF density fit on a single class; early-stop on that class's val NLL."""
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=len(FLOW_COLS), context=0, transforms=N_TRANSFORMS,
                          bins=N_BINS, hidden_features=HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(flow.parameters(), lr=LR,
                            fused=(DEVICE.type == "cuda"))
    Xtr = torch.as_tensor(Xtr_c, device=DEVICE)
    Xva = torch.as_tensor(Xva_c, device=DEVICE)
    best_nll, best_state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        flow.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s: s + BATCH]
            loss = -flow().log_prob(Xtr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            vnll = float(-flow().log_prob(Xva).mean().item())
        if vnll < best_nll - 1e-4:
            best_nll = vnll
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    flow.load_state_dict(best_state)
    flow.eval()
    del Xtr, Xva
    return flow, best_nll


@torch.no_grad()
def log_density(flow, X_std: np.ndarray) -> np.ndarray:
    out = np.empty(len(X_std), dtype=np.float64)
    for s in range(0, len(X_std), EVAL_BS):
        xb = torch.as_tensor(X_std[s: s + EVAL_BS], device=DEVICE)
        out[s: s + EVAL_BS] = flow().log_prob(xb).double().cpu().numpy()
    return out


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-logdens", action="store_true",
                    help="also save raw log p(x|class) arrays for density-augmented base")
    args = ap.parse_args()

    train = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    test = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    y = np.array([CLASS_MAP[c] for c in train[TARGET].to_numpy()], dtype=np.int64)
    test_ids = test[ID].to_numpy()
    X = to_matrix(train)
    X_test = to_matrix(test)
    print(f"X {X.shape}  flow cols {FLOW_COLS}  device {DEVICE}")
    print(f"NSF transforms={N_TRANSFORMS} bins={N_BINS} hidden={HIDDEN}  bs={BATCH} lr={LR}")
    if args.save_logdens:
        print("  --save-logdens: will also save raw log p(x|class) (no prior, no softmax)")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    test_proba = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    oof_logdens = np.zeros((len(X), len(CLASSES)), dtype=np.float64) if args.save_logdens else None
    test_logdens = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64) if args.save_logdens else None
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler().fit(X[tri])
        Xtr_std = scaler.transform(X[tri]).astype(np.float32)
        Xva_std = scaler.transform(X[vai]).astype(np.float32)
        Xte_std = scaler.transform(X_test).astype(np.float32)
        log_prior = np.log(np.bincount(y[tri], minlength=len(CLASSES)) / len(tri))

        val_scores = np.zeros((len(vai), len(CLASSES)), dtype=np.float64)
        test_scores = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
        val_raw = np.zeros((len(vai), len(CLASSES)), dtype=np.float64)
        test_raw = np.zeros((len(X_test), len(CLASSES)), dtype=np.float64)
        nlls = []
        for c in range(len(CLASSES)):
            Xc = Xtr_std[y[tri] == c]
            Xvc = Xva_std[y[vai] == c]
            flow, nll = train_class_flow(Xc, Xvc, SEED + fold * 10 + c)
            nlls.append(round(nll, 3))
            raw_val = log_density(flow, Xva_std)
            raw_test = log_density(flow, Xte_std)
            val_scores[:, c] = raw_val + log_prior[c]
            test_scores[:, c] = raw_test + log_prior[c]
            val_raw[:, c] = raw_val
            test_raw[:, c] = raw_test
            del flow
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        oof[vai] = softmax_rows(val_scores)
        test_proba += softmax_rows(test_scores) / FOLDS
        if args.save_logdens:
            oof_logdens[vai] = val_raw
            test_logdens += test_raw / FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_scores, axis=1))))
        rec_f = recall_score(y[vai], np.argmax(val_scores, axis=1), average=None, labels=[0, 1, 2])
        print(f"  Fold {fold} bal_acc {fold_scores[-1]:.5f}  recall(G,Q,S) {rec_f.round(4)}  "
              f"val_nll(G,Q,S) {nlls}")

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF bal_acc (argmax): {argmax:.5f}  [generative ~0.95-0.96 EXPECTED; decorrelation is the point]")
    print(f"per-class recall {dict(zip(CLASSES, rec.round(4)))}")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF bal_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, CLASSES, RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    if args.save_logdens and oof_logdens is not None and test_logdens is not None:
        np.save(RESULTS_DIR / f"oof_{RUN}_logdens.npy", oof_logdens.astype(np.float32))
        np.save(RESULTS_DIR / f"test_{RUN}_logdens.npy", test_logdens.astype(np.float32))
        print(f"Saved raw log-density arrays → oof_{RUN}_logdens.npy, test_{RUN}_logdens.npy")
    labels = [CLASSES[i] for i in np.argmax(test_proba * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pl.DataFrame({ID: test_ids, TARGET: labels}).write_csv(SUBMISSIONS_DIR / f"{RUN}.csv")
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
