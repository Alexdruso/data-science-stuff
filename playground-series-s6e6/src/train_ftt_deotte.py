"""From-scratch FT-Transformer + logit-adjust — a NEW strong diverse NN family (S6E6 campaign).

Library NNs (pytabkit TabM/FTT/RealMLP_TD) all STAR-wall here because they have no loss hook for
the logit-adjustment that is the minority fix. So we build FT-Transformer (Gorishniy 2021) from
scratch and inject logit-adjustment (add tau·log(prior) to logits in the TRAIN loss only; raw at
inference) — the exact lever that made our from-scratch RealMLP work. Attention over feature tokens
is a fundamentally different inductive bias than RealMLP/GBDT → maximal DECORRELATION (the only
thing the stacker rewards). Reuses the RealMLP modest FE (32 tokens) + per-fold TE on the 2 combo
cats; id-sorted I/O; GPU-mem rule (del + empty_cache per fold).

GATE: standalone ~0.965+ AND decorrelated from realmlp_deotte (→ lifts the stack). Watch STAR recall
(logit-adjust should keep it off the 0.91 wall).

Run:  python src/train_ftt_deotte.py
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from cv_results import save_cv_result
from postprocess import optimize_thresholds, save_threshold_weights
from train_realmlp_deotte import CLASS_MAP, ID, INV_CLASS_MAP, TARGET, feature_engineering

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CKPT = RESULTS_DIR / "_ftt_ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, FOLDS, N_CLASSES = 42, 5, 3

D_TOKEN, N_LAYERS, N_HEADS = 192, 3, 8
DROPOUT, ATTN_DROPOUT = 0.1, 0.2
LR, WD, BATCH, EPOCHS, PATIENCE = 1e-3, 1e-5, 1024, 60, 12
TAU, LS_EPS = 1.075, 0.04   # logit-adjust power + label smoothing
RUN = "ftt_deotte"


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_dims: list[int]) -> None:
        super().__init__()
        self.n_num = n_num
        self.num_w = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.num_b = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.cat_embs = nn.ModuleList([nn.Embedding(d, D_TOKEN) for d in cat_dims])
        self.cls = nn.Parameter(torch.randn(1, 1, D_TOKEN) * 0.02)
        layer = nn.TransformerEncoderLayer(
            D_TOKEN, N_HEADS, dim_feedforward=int(D_TOKEN * 4 / 3), dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True)
        layer.self_attn.dropout = ATTN_DROPOUT
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D_TOKEN), nn.GELU(), nn.Linear(D_TOKEN, N_CLASSES))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = [self.cls.expand(x_num.shape[0], -1, -1),
                x_num.unsqueeze(-1) * self.num_w + self.num_b]
        if self.cat_embs:
            toks.append(torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1))
        z = self.encoder(torch.cat(toks, dim=1))
        return self.head(z[:, 0])


def train_fold(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, Xn_te, Xc_te, cat_dims, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FTTransformer(Xn_tr.shape[1], cat_dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    log_prior = np.log(counts) - np.log(counts).mean()
    adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

    Xn = torch.tensor(Xn_tr, dtype=torch.float32, device=DEVICE)
    Xc = torch.tensor(Xc_tr, dtype=torch.long, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    vn, vc = torch.tensor(Xn_va, dtype=torch.float32, device=DEVICE), torch.tensor(Xc_va, dtype=torch.long, device=DEVICE)
    tn, tc = torch.tensor(Xn_te, dtype=torch.float32, device=DEVICE), torch.tensor(Xc_te, dtype=torch.long, device=DEVICE)

    def predict(xn, xc):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(xn), 8192):
                out.append(torch.softmax(model(xn[s:s+8192], xc[s:s+8192]), 1).cpu().numpy())
        return np.concatenate(out)

    best_ba, wait, n = -1.0, 0, len(Xn)
    best_val = best_test = None
    for _ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = crit(model(Xn[idx], Xc[idx]) + adj, yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        vp = predict(vn, vc)
        ba = balanced_accuracy_score(y_va, vp.argmax(1))
        if ba > best_ba:
            best_ba, wait, best_val = ba, 0, vp
            best_test = predict(tn, tc)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    del model, opt, sched, Xn, Xc, yt, vn, vc, tn, tc
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_val, best_test, best_ba


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    print(f"Device {DEVICE}  d_token={D_TOKEN} L={N_LAYERS} TAU={TAU}")
    train = pd.read_csv(DATA_DIR / "train.csv").sort_values(ID).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv").sort_values(ID).reset_index(drop=True)
    train[TARGET] = train[TARGET].map(CLASS_MAP)
    X = train.drop([ID, TARGET], axis=1); y = train[TARGET]
    X_test = test.drop([ID], axis=1); test_ids = test[ID].to_numpy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    cmap: dict = {}
    X, new_cat, _, _ = feature_engineering(X, cat_cols, num_cols, cmap, fit=True)
    X_test, _, new_num, combos = feature_engineering(X_test, cat_cols, num_cols, cmap, fit=False)
    cat_cols = sorted(cat_cols + new_cat); num_cols += new_num
    X = X.reindex(sorted(X.columns), axis=1); X_test = X_test.reindex(sorted(X_test.columns), axis=1)
    num_cols = sorted([c for c in num_cols if c in X.columns])
    print(f"X {X.shape}  num {len(num_cols)}  cat {len(cat_cols)}")

    yv = y.to_numpy()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    test_proba = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, yv), 1):
        X_tr, X_va, X_te = X.iloc[tri].copy(), X.iloc[vai].copy(), X_test.copy()
        y_tr = yv[tri]
        # per-fold TE on combos → numeric tokens
        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED + fold)
        te_names = [f"_{c}TE_class{k}" for c in combos for k in range(N_CLASSES)]
        X_tr[te_names] = enc.fit_transform(X_tr[combos], y_tr).astype(np.float32)
        X_va[te_names] = enc.transform(X_va[combos]).astype(np.float32)
        X_te[te_names] = enc.transform(X_te[combos]).astype(np.float32)
        numc = sorted(set(num_cols) | set(te_names))
        # numeric: quantile-normal transform (fit on train fold)
        qt = QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=SEED)
        Xn_tr = qt.fit_transform(X_tr[numc].to_numpy(np.float32)).astype(np.float32)
        Xn_va = qt.transform(X_va[numc].to_numpy(np.float32)).astype(np.float32)
        Xn_te = qt.transform(X_te[numc].to_numpy(np.float32)).astype(np.float32)
        # cats: int codes, dims from all splits, clamp unseen
        cat_dims, Xc_tr, Xc_va, Xc_te = [], [], [], []
        for c in cat_cols:
            tr_c = X_tr[c].astype(str); va_c = X_va[c].astype(str); te_c = X_te[c].astype(str)
            cats = pd.Index(sorted(set(tr_c) | set(va_c) | set(te_c)))
            m = {v: i for i, v in enumerate(cats)}
            cat_dims.append(len(cats))
            Xc_tr.append(tr_c.map(m).to_numpy()); Xc_va.append(va_c.map(m).to_numpy()); Xc_te.append(te_c.map(m).to_numpy())
        Xc_tr = np.stack(Xc_tr, 1) if cat_dims else np.zeros((len(X_tr), 0), int)
        Xc_va = np.stack(Xc_va, 1) if cat_dims else np.zeros((len(X_va), 0), int)
        Xc_te = np.stack(Xc_te, 1) if cat_dims else np.zeros((len(X_te), 0), int)

        vp, tp, ba = train_fold(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, yv[vai], Xn_te, Xc_te, cat_dims, SEED + fold * 100)
        oof[vai] = vp; test_proba += tp / FOLDS
        fold_scores.append(float(balanced_accuracy_score(yv[vai], vp.argmax(1))))
        if fold == 1:
            print(f"  n_num_tokens {len(numc)}  n_cat_tokens {len(cat_dims)}")
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.5f}")

    pred = oof.argmax(1)
    argmax = float(balanced_accuracy_score(yv, pred))
    rec = recall_score(yv, pred, average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.5f}  [realmlp_deotte 0.96888; TabM walled STAR 0.909]")
    print(f"per-class recall {dict(zip(['GALAXY','QSO','STAR'], rec.round(4)))}  [watch STAR off the 0.91 wall]")
    tw, best = optimize_thresholds(oof, yv)
    print(f"OOF balanced_acc (threshold-tuned): {best:.5f}")

    save_threshold_weights(tw, ["GALAXY", "QSO", "STAR"], RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = [INV_CLASS_MAP[i] for i in np.argmax(test_proba * tw, axis=1)]
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID: test_ids, TARGET: labels}).to_csv(SUBMISSIONS_DIR / f"{RUN}.csv", index=False)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
