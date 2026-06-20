"""Denoising-autoencoder representation base (Jahrer's Porto Seguro trick) — diversity bet (S6E6).

The in-house reshuffle space is tapped at ~0.9698: every base we add either ranks worse or is
redundant. A DAE learns a NEW nonlinear representation of the SAME columns — the one well-known
architecture that can (in principle) expose split directions our raw-feature bases can't axis-align.

Recipe (Jahrer 2017):
  1. Train UNSUPERVISED on train+test features together (no labels -> no target leakage; AUC=0.5
     means same distribution, so the test features are clean transductive signal).
  2. SWAP NOISE corruption: per column, with prob p replace a value by another random row's value
     from the same column (corrupt within the empirical marginal). Reconstruct the CLEAN input.
  3. Representation = concatenated encoder hidden activations (overcomplete -> rich even from few
     inputs). Feed to a downstream LGBM base, honest 5-fold (DAE never saw labels -> OOF clean).

Reports standalone balanced-acc AND stack delta vs the 7-base LR stack, for DAE-only and raw+DAE.
GPU-memory rule: del model+tensors, empty_cache after embedding extraction.

Run:  python src/train_dae.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from features import TARGET, build_features
from lgbm_device import get_lgbm_device

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
STACK_MODELS = ["lgbm", "xgboost", "catboost", "lgbm_fe",
                "xgb_deotte", "realmlp_deotte", "catboost_deotte"]
NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift",
            "u_g", "g_r", "r_i", "i_z", "u_z", "log1p_redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
EPS = 1e-6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SWAP_P = 0.15
HID = 256
EPOCHS = 25
BS = 4096


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS))


class DAE(nn.Module):
    def __init__(self, d_in, h=HID):
        super().__init__()
        self.e1, self.e2, self.e3 = nn.Linear(d_in, h), nn.Linear(h, h), nn.Linear(h, h)
        self.d1, self.d2 = nn.Linear(h, h), nn.Linear(h, d_in)

    def encode(self, x):
        h1 = F.relu(self.e1(x))
        h2 = F.relu(self.e2(h1))
        h3 = F.relu(self.e3(h2))
        return h1, h2, h3

    def forward(self, x):
        _, _, h3 = self.encode(x)
        return self.d2(F.relu(self.d1(h3)))

    def represent(self, x):
        h1, h2, h3 = self.encode(x)
        return torch.cat([h1, h2, h3], dim=1)


def swap_noise(x, p):
    """Per-column swap: with prob p, replace a value by another random row's value (same column)."""
    b, d = x.shape
    mask = torch.rand(b, d, device=x.device) < p
    idx = torch.randint(0, b, (b, d), device=x.device)
    cols = torch.arange(d, device=x.device).unsqueeze(0).expand(b, d)
    return torch.where(mask, x[idx, cols], x)


def build_input(tr, te):
    """Standardized numerics + one-hot cats, fit on train+test (unsupervised). Returns float32."""
    num = np.vstack([tr.select(NUM_COLS).to_numpy(), te.select(NUM_COLS).to_numpy()]).astype(np.float64)
    num = StandardScaler().fit_transform(num)
    oh = []
    for c in CAT_COLS:
        vals = np.concatenate([tr[c].to_numpy(), te[c].to_numpy()])
        for v in sorted(set(vals.tolist())):
            oh.append((vals == v).astype(np.float64))
    X = np.hstack([num, np.vstack(oh).T]).astype(np.float32)
    return X[: len(tr)], X[len(tr):]


def train_dae(X_all):
    torch.manual_seed(42)
    model = DAE(X_all.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    data = torch.tensor(X_all, device=DEV)
    n = len(data)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(0, n, BS):
            xb = data[perm[s:s + BS]]
            opt.zero_grad()
            loss = F.mse_loss(model(swap_noise(xb, SWAP_P)), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  DAE epoch {ep:2d}  recon-mse {tot / n:.5f}")
    del data, opt
    return model


def embed(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 16384):
            xb = torch.tensor(X[s:s + 16384], device=DEV)
            out.append(model.represent(xb).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def stack_delta(extra_oof, y):
    base_tr = [logit(np.load(RESULTS_DIR / f"oof_{m}.npy")) for m in STACK_MODELS]

    def run(blocks):
        Z = np.hstack(blocks)
        oof = np.zeros((len(y), 3))
        skf = StratifiedKFold(5, shuffle=True, random_state=2024)
        for tri, vai in skf.split(Z, y):
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            lr.fit(Z[tri], y[tri])
            oof[vai] = lr.predict_proba(Z[vai])
        return float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))

    return run(base_tr), run(base_tr + [logit(extra_oof)])


def lgbm_oof(X, y, dev_type, n_jobs):
    oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        m = LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                           n_estimators=250, learning_rate=0.05, num_leaves=31,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.4, max_bin=127, reg_lambda=1.0, random_state=42,
                           verbose=-1, n_jobs=n_jobs, device_type=dev_type)
        m.fit(X[tri], y[tri])
        oof[vai] = m.predict_proba(X[vai])
        print(f"    fold {k} done", flush=True)
    return oof


def main() -> None:
    dev_type, n_jobs = get_lgbm_device()
    tr = build_features(pl.read_csv(DATA_DIR / "train.csv"))
    te = build_features(pl.read_csv(DATA_DIR / "test.csv"))
    le = LabelEncoder()
    y = le.fit_transform(tr[TARGET].to_numpy())

    Xtr_raw, Xte_raw = build_input(tr, te)
    print(f"DAE input dim {Xtr_raw.shape[1]}; training on {len(Xtr_raw) + len(Xte_raw)} rows ({DEV})")
    model = train_dae(np.vstack([Xtr_raw, Xte_raw]))
    Etr, Ete = embed(model, Xtr_raw), embed(model, Xte_raw)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    print(f"embeddings: {Etr.shape[1]}-dim\n")

    variants = {
        "dae": Etr,
        "dae_plus_raw": np.hstack([Etr, Xtr_raw]),
    }
    for name, X in variants.items():
        oof = lgbm_oof(X, y, dev_type, n_jobs)
        am = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
        rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
        base, withx = stack_delta(oof, y)
        print(f"{name:13s} standalone {am:.4f}  rec {dict(zip(le.classes_, rec.round(3)))}")
        print(f"{' '*13} stack 7-base {base:.5f} -> +{name} {withx:.5f}  (delta {withx - base:+.5f})\n")
        np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
        save_cv_result(RESULTS_DIR, name, [], am, metric_name="balanced_acc")


if __name__ == "__main__":
    main()
