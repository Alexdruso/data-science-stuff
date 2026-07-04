"""MLP with LOGIT-ADJUSTED loss (balanced softmax) — the NN STAR fix (PS S6E6).

cdeotte's RealMLP reaches CV 0.9688 (strong + diverse) where ours/TabM/TabICL all
stall at ~0.95 with STAR recall ~0.91. The subagent extraction found the lever: NOT
class weights (he turned them OFF, as we found), NOT oversampling (OFF), but
`loss_prior_power=1.075` — balanced-softmax / logit adjustment. During training add
tau·log(prior_c) to the class logits before cross-entropy; predict with RAW logits.
This shifts the decision boundary toward rare STAR without distorting inference
probabilities — exactly the STAR fix our plain-CE MLP (0.9547) lacks.

We add it to our own PyTorch MLP (which already early-stops on val balanced acc) —
the cheapest test of the lever before porting his full custom architecture. No
class weight, no oversampling (both confirmed-flat). Watch STAR recall.

Run:  python src/train_mlp_la.py
"""

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
from train_mlp import (
    BATCH_SIZE, DEFAULT_PARAMS, DEVICE, EPOCHS, MLP, N_CLASSES, N_FOLDS, PATIENCE,
    build_layer_sizes, prepare_arrays,
)

from data_science_stuff.kaggle.io import competition_dirs, write_submission

DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs(__file__)
TAU = 1.075   # cdeotte's loss_prior_power
RUN = "mlp_la"


def train_fold(X_tr, y_tr, X_val, y_val, X_test, num_idx, cat_idx, params):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    sc = StandardScaler()
    X_tr_s, X_val_s, X_test_s = (np.empty_like(a) for a in (X_tr, X_val, X_test))
    X_tr_s[:, num_idx] = pt.fit_transform(X_tr[:, num_idx])
    X_val_s[:, num_idx] = pt.transform(X_val[:, num_idx])
    X_test_s[:, num_idx] = pt.transform(X_test[:, num_idx])
    X_tr_s[:, cat_idx] = sc.fit_transform(X_tr[:, cat_idx])
    X_val_s[:, cat_idx] = sc.transform(X_val[:, cat_idx])
    X_test_s[:, cat_idx] = sc.transform(X_test[:, cat_idx])

    model = MLP(X_tr_s.shape[1], build_layer_sizes(params), float(params["dropout"]),
                params["activation"] == "gelu").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]),
                                  weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # logit adjustment offset: tau * log(prior), prior geomean-normalized (Deotte)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    log_prior = np.log(counts) - np.log(counts).mean()
    adj = torch.tensor(TAU * log_prior, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss()  # NO class weight (Deotte off)

    X_tr_t = torch.from_numpy(X_tr_s).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
    X_val_t = torch.from_numpy(X_val_s).to(DEVICE)
    X_test_t = torch.from_numpy(X_test_s).to(DEVICE)

    best_ba, patience_counter, n = -1.0, 0, len(X_tr_t)
    best_val_pred = np.zeros((len(X_val_s), N_CLASSES))
    best_test_pred = np.zeros((len(X_test_s), N_CLASSES))
    for _epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            optimizer.zero_grad()
            # adjusted logits in the loss; raw logits at inference
            loss = criterion(model(X_tr_t[idx]) + adj, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_proba = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
        ba = float(balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1)))
        if ba > best_ba:
            best_ba, patience_counter, best_val_pred = ba, 0, val_proba
            with torch.no_grad():
                best_test_pred = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    del model, optimizer, scheduler, criterion, X_tr_t, y_tr_t, X_val_t, X_test_t
    return best_val_pred, best_test_pred


def main() -> None:
    print(f"Device: {DEVICE}  TAU={TAU}")
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
    print(f"X {X.shape}  classes {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X), N_CLASSES))
    test_proba = np.zeros((len(X_test), N_CLASSES))
    fold_scores = []
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        val_pred, test_pred = train_fold(X[tri], y[tri], X[vai], y[vai], X_test,
                                         num_idx, cat_idx, DEFAULT_PARAMS)
        oof[vai] = val_pred
        test_proba += test_pred / N_FOLDS
        fold_scores.append(float(balanced_accuracy_score(y[vai], np.argmax(val_pred, axis=1))))
        print(f"  Fold {fold} balanced_acc: {fold_scores[-1]:.4f}")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    argmax = float(balanced_accuracy_score(y, np.argmax(oof, axis=1)))
    rec = recall_score(y, np.argmax(oof, axis=1), average=None, labels=[0, 1, 2])
    print(f"\nOOF balanced_acc (argmax): {argmax:.4f}  [plain MLP 0.9547; deotte-realmlp 0.9688]")
    print(f"per-class recall {dict(zip(le.classes_, rec.round(4)))}  [watch STAR vs plain-MLP 0.958]")
    tw, best = optimize_thresholds(oof, y)
    print(f"OOF balanced_acc (threshold-tuned): {best:.4f}")

    save_threshold_weights(tw, le.classes_.tolist(), RESULTS_DIR / f"threshold_weights_{RUN}.json")
    save_cv_result(RESULTS_DIR, RUN, fold_scores, best, metric_name="balanced_acc")
    np.save(RESULTS_DIR / f"oof_{RUN}.npy", oof)
    np.save(RESULTS_DIR / f"test_{RUN}.npy", test_proba)
    labels = le.inverse_transform(np.argmax(test_proba * tw, axis=1))
    write_submission(SUBMISSIONS_DIR, f"{RUN}.csv", test_ids, TARGET, labels)
    print(f"Saved → {RUN}")


if __name__ == "__main__":
    main()
