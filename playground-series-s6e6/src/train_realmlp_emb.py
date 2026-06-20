"""MLP with PLR numeric embeddings + logit-adjusted loss — the NN frontier test (S6E6).

The binding constraint is "no strong DIVERSE base" for the LR stacker. Every other lever
is flat. cdeotte's RealMLP reaches CV 0.9688 where all our NNs stall ~0.95 with STAR ~0.91.
We've tested both halves of his recipe SEPARATELY and both were flat:
  - logit-adjust alone (mlp_la 0.9548): slides STAR↔GALAXY along the frontier, net 0.
  - PBLD embeddings alone (pytabkit realmlp 0.9508): STAR 0.93, intrinsic without a loss lever.
The ONLY untested variable is the two TOGETHER — the hypothesis that the loss lever can
reach the minority boundary that the embedding architecture alone cannot.

PLR = Periodic-Linear-ReLU numeric embedding (Gorishniy 2022 "On Embeddings for Numerical
Features"), the proven-strong variant: per-feature learnable frequencies → sin/cos →
per-feature linear → ReLU. This lets the net carve the curved low-z manifold that axis-
aligned splits and a plain linear input layer both miss.

GATE = FRONTIER SHIFT, not aggregate balanced-acc (logit-adjust only slides along the
frontier; only the embedding can push it OUT). Beat our best NN point (realmlp: GALAXY
0.963 / STAR 0.93) on BOTH classes: STAR → 0.95+ while GALAXY holds ≥ 0.96 = frontier moved
→ scale to full RealMLP (n_ens=8, EMA, NTP) then re-stack. STAR only bought by giving back
GALAXY = same wall → NN branch closed, 0.9662 is the honest finish.

Single net (n_ens=1) deliberately: the expensive machinery lifts the LEVEL, not the frontier
SHAPE, so it can't rescue a flat core. No throwaway code — this core IS the start of the build.

Run:  python src/train_realmlp_emb.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import EXCLUDE_COLS, TARGET, build_features, compute_group_features
from postprocess import optimize_thresholds, save_threshold_weights
from train_mlp import N_CLASSES, N_FOLDS, prepare_arrays

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = "realmlp_emb"

# Embedding + training config
K_FREQ = 24          # learnable periodic frequencies per numeric feature
D_EMB = 8            # embedding width per numeric feature
EMB_SIGMA = 0.1      # frequency init scale (Gorishniy default ~0.01-0.1)
LAYERS = [512, 256, 128]
DROPOUT = 0.15
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
EPOCHS = 150
PATIENCE = 15
TAU = 1.075          # cdeotte's loss_prior_power (logit adjustment)


class PLREmbedding(nn.Module):
    """Per-feature periodic→linear→ReLU embedding for numeric inputs."""

    def __init__(self, n_num: int, k: int, d_emb: int, sigma: float) -> None:
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(n_num, k) * sigma)
        self.weight = nn.Parameter(torch.randn(n_num, 2 * k, d_emb) / math.sqrt(2 * k))
        self.bias = nn.Parameter(torch.zeros(n_num, d_emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n_num)
        v = 2.0 * math.pi * x.unsqueeze(-1) * self.coeffs            # (B, n_num, k)
        p = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)         # (B, n_num, 2k)
        e = torch.einsum("bnk,nkd->bnd", p, self.weight) + self.bias  # (B, n_num, d_emb)
        return torch.relu(e).flatten(1)                             # (B, n_num*d_emb)


class EmbMLP(nn.Module):
    def __init__(self, n_num: int, n_cat: int) -> None:
        super().__init__()
        self.emb = PLREmbedding(n_num, K_FREQ, D_EMB, EMB_SIGMA)
        in_size = n_num * D_EMB + n_cat
        blocks: list[nn.Module] = []
        for size in LAYERS:
            blocks += [nn.Linear(in_size, size), nn.BatchNorm1d(size),
                       nn.GELU(), nn.Dropout(DROPOUT)]
            in_size = size
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(in_size, N_CLASSES)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.emb(x_num), x_cat], dim=1) if x_cat.shape[1] else self.emb(x_num)
        return self.head(self.body(h))


def train_fold(X_tr, y_tr, X_val, y_val, X_test, num_idx, cat_idx):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_tr_s, X_val_s, X_test_s = (np.empty_like(a) for a in (X_tr, X_val, X_test))
    X_tr_s[:, num_idx] = pt.fit_transform(X_tr[:, num_idx])
    X_val_s[:, num_idx] = pt.transform(X_val[:, num_idx])
    X_test_s[:, num_idx] = pt.transform(X_test[:, num_idx])
    if cat_idx:
        sc = StandardScaler()
        X_tr_s[:, cat_idx] = sc.fit_transform(X_tr[:, cat_idx])
        X_val_s[:, cat_idx] = sc.transform(X_val[:, cat_idx])
        X_test_s[:, cat_idx] = sc.transform(X_test[:, cat_idx])

    def split(a: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(a).to(DEVICE)
        return t[:, num_idx], (t[:, cat_idx] if cat_idx else t[:, :0])

    Xn_tr, Xc_tr = split(X_tr_s)
    Xn_val, Xc_val = split(X_val_s)
    Xn_te, Xc_te = split(X_test_s)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)

    model = EmbMLP(len(num_idx), len(cat_idx)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    log_prior = np.log(counts) - np.log(counts).mean()
    adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss()  # NO class weight (Deotte off); logit-adjust instead

    best_ba, patience_counter, n = -1.0, 0, len(Xn_tr)
    best_val = np.zeros((len(X_val_s), N_CLASSES))
    best_test = np.zeros((len(X_test_s), N_CLASSES))
    for _epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xn_tr[idx], Xc_tr[idx]) + adj, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_proba = torch.softmax(model(Xn_val, Xc_val), dim=1).cpu().numpy()
        ba = float(balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1)))
        if ba > best_ba:
            best_ba, patience_counter, best_val = ba, 0, val_proba
            with torch.no_grad():
                best_test = torch.softmax(model(Xn_te, Xc_te), dim=1).cpu().numpy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    del model, optimizer, scheduler, criterion
    del Xn_tr, Xc_tr, Xn_val, Xc_val, Xn_te, Xc_te, y_tr_t
    return best_val, best_test


def main() -> None:
    print(f"Device: {DEVICE}  TAU={TAU}  K={K_FREQ} D_EMB={D_EMB}")
    train_raw = pl.read_csv(DATA_DIR / "train.csv")
    test_raw = pl.read_csv(DATA_DIR / "test.csv")
    train_pl = compute_group_features(train_raw, build_features(train_raw))
    test_pl = compute_group_features(train_raw, build_features(test_raw))
    cat_cols = [c for c in train_pl.columns
                if train_pl[c].dtype == pl.String and c not in EXCLUDE_COLS]
    feature_cols = [c for c in train_pl.columns if c not in EXCLUDE_COLS]
    X, X_test, num_idx, cat_idx = prepare_arrays(train_pl, test_pl, cat_cols, feature_cols)
    le = LabelEncoder()
    y = le.fit_transform(train_pl.to_pandas()[TARGET].to_numpy())
    test_ids = test_pl.to_pandas()["id"].to_numpy()
    print(f"X {X.shape}  n_num {len(num_idx)}  n_cat {len(cat_idx)}  classes {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        val_pred, test_pred = train_fold(X[tri], y[tri], X[vai], y[vai], X_test, num_idx, cat_idx)
        oof[vai] = val_pred
        test_proba += test_pred / N_FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_pred, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    pred = np.argmax(oof, axis=1)
    argmax = float(balanced_accuracy_score(y, pred))
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    recd = dict(zip(le.classes_, rec.round(4)))
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [mlp_la 0.9548; realmlp 0.9508; deotte 0.9688]")
    print(f"per-class recall {recd}")
    print(f"FRONTIER GATE — beat realmlp point (GALAXY 0.963 / STAR 0.93) on BOTH: "
          f"GALAXY {recd.get('GALAXY')} (≥0.96?) STAR {recd.get('STAR')} (→0.95+?)")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
