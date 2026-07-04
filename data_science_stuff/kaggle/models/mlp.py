"""Baseline tabular MLP: the standard non-GBDT diversity base.

Extracted from the s6e6 ``train_mlp.py`` template (itself adapted from
s6e5's binary MLP). :func:`fit_mlp_fold` is a drop-in ``fit_fold`` body for
:func:`data_science_stuff.kaggle.cv.run_cv`: per-fold Yeo-Johnson/standard
scaling fit on the training split only, early stopping on a validation
score, optional class weights or logit adjustment for imbalanced targets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch import nn

from data_science_stuff.kaggle.blending import ScoreFn


def build_layer_sizes(
    n_layers: int, first_size: int, shrink: float = 0.5, *, min_size: int = 32
) -> list[int]:
    """Geometrically shrinking hidden-layer sizes, floored at ``min_size``.

    >>> build_layer_sizes(3, 512)
    [512, 256, 128]
    """
    sizes = [first_size]
    for _ in range(n_layers - 1):
        sizes.append(max(min_size, int(sizes[-1] * shrink)))
    return sizes


class MLP(nn.Module):
    """Plain feed-forward classifier: Linear → BatchNorm → activation → Dropout."""

    def __init__(
        self,
        n_features: int,
        layer_sizes: Sequence[int],
        dropout: float,
        *,
        n_classes: int,
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        act_cls: type[nn.Module] = nn.GELU if activation == "gelu" else nn.ReLU
        blocks: list[nn.Module] = []
        in_size = n_features
        for size in layer_sizes:
            blocks += [
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),
                act_cls(),
                nn.Dropout(dropout),
            ]
            in_size = size
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(in_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (apply softmax outside)."""
        out: torch.Tensor = self.output_layer(self.body(x))
        return out


def class_weights(y: NDArray[np.int64], n_classes: int) -> NDArray[np.float32]:
    """Inverse-frequency weights, sklearn 'balanced' formula: n / (k * count)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return np.asarray(len(y) / (n_classes * counts), dtype=np.float32)


def ohe_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: Sequence[str],
    feature_cols: Sequence[str],
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[int], list[int]]:
    """One-hot encode categoricals over train+test; split back into matrices.

    Encoding categories over the concatenation is leak-free (it uses no
    targets) and guarantees train/test share the same columns even when a
    category is missing from one split.

    Returns:
        ``(X, X_test, num_idx, cat_idx)`` — float32 matrices with numeric
        columns first, plus the column indices of each block (feed them to
        :func:`fit_mlp_fold` so scaling can treat the blocks differently).
    """
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]

    combined = pd.concat(
        [train[list(cat_cols)], test[list(cat_cols)]], ignore_index=True
    )
    ohe = pd.get_dummies(combined, columns=list(cat_cols), dtype=np.float32)
    n_train = len(train)
    train_ohe = ohe.iloc[:n_train].to_numpy(dtype=np.float32)
    test_ohe = ohe.iloc[n_train:].to_numpy(dtype=np.float32)

    x_num = train[num_cols].to_numpy(dtype=np.float32)
    x_test_num = test[num_cols].to_numpy(dtype=np.float32)

    x = np.hstack([x_num, train_ohe])
    x_test = np.hstack([x_test_num, test_ohe])
    n_num = len(num_cols)
    num_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_num + train_ohe.shape[1]))
    return x, x_test, num_idx, cat_idx


def _default_score(y_val: NDArray[Any], proba: NDArray[np.float64]) -> float:
    return float(balanced_accuracy_score(y_val, np.argmax(proba, axis=1)))


def fit_mlp_fold(
    X_tr: NDArray[np.float32],
    y_tr: NDArray[np.int64],
    X_va: NDArray[np.float32],
    y_va: NDArray[np.int64],
    X_test: NDArray[np.float32],
    num_idx: Sequence[int],
    cat_idx: Sequence[int],
    params: Mapping[str, Any],
    *,
    n_classes: int,
    device: torch.device | None = None,
    epochs: int = 100,
    batch_size: int = 8192,
    patience: int = 10,
    use_class_weights: bool = True,
    logit_adjust: torch.Tensor | None = None,
    score_fn: ScoreFn | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Train one MLP fold and return ``(val_proba, test_proba)``.

    Per-fold preprocessing is fit on the training split only (no leakage):
    Yeo-Johnson on numeric columns, StandardScaler on the OHE block. Early
    stopping tracks ``score_fn`` on the validation split (default: balanced
    accuracy on argmax) and returns the best epoch's predictions.

    For imbalanced targets pick ONE lever: ``use_class_weights=True``
    (inverse-frequency CE weights), or pass ``logit_adjust`` from
    :func:`data_science_stuff.kaggle.models.losses.logit_adjustment`
    (added to logits in the loss, raw logits at inference) with
    ``use_class_weights=False`` — s6e6 found the combination redundant.

    GPU tensors and the model are deleted before returning (GPU memory
    rule); the caller's fold loop must still run ``torch.cuda.empty_cache()``
    after each fold (e.g. via ``run_cv``'s ``after_fold`` hook).

    Args:
        X_tr: Training features (float32, numeric block then OHE block).
        y_tr: Training labels.
        X_va: Validation features.
        y_va: Validation labels.
        X_test: Test features (predicted at the best epoch).
        num_idx: Column indices of the numeric block.
        cat_idx: Column indices of the OHE block.
        params: ``n_layers``, ``first_size``, ``shrink``, ``dropout``,
            ``lr``, ``weight_decay``, ``activation`` — the shape written by
            an Optuna ``tune_mlp.py`` and read via ``load_params``.
        n_classes: Output dimension.
        device: Defaults to CUDA when available.
        epochs: Maximum epochs.
        batch_size: Minibatch size.
        patience: Early-stopping patience in epochs.
        use_class_weights: Weight the CE loss by inverse class frequency.
        logit_adjust: Optional (n_classes,) logit-adjustment offsets.
        score_fn: Early-stopping metric ``(y_va, val_proba) -> float``.

    Returns:
        ``(val_proba, test_proba)`` from the best-scoring epoch.
    """
    dev = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    score = score_fn if score_fn is not None else _default_score

    # Per-fold scaling fit on the train split only (no leakage).
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    sc = StandardScaler()
    x_tr_s, x_va_s, x_test_s = (np.empty_like(a) for a in (X_tr, X_va, X_test))
    num_list, cat_list = list(num_idx), list(cat_idx)
    x_tr_s[:, num_list] = pt.fit_transform(X_tr[:, num_list])
    x_va_s[:, num_list] = pt.transform(X_va[:, num_list])
    x_test_s[:, num_list] = pt.transform(X_test[:, num_list])
    if cat_list:
        x_tr_s[:, cat_list] = sc.fit_transform(X_tr[:, cat_list])
        x_va_s[:, cat_list] = sc.transform(X_va[:, cat_list])
        x_test_s[:, cat_list] = sc.transform(X_test[:, cat_list])

    layer_sizes = build_layer_sizes(
        int(params["n_layers"]),
        int(params["first_size"]),
        float(params.get("shrink", 0.5)),
    )
    model = MLP(
        x_tr_s.shape[1],
        layer_sizes,
        float(params["dropout"]),
        n_classes=n_classes,
        activation="gelu" if params.get("activation") == "gelu" else "relu",
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if use_class_weights:
        weight_t = torch.from_numpy(class_weights(y_tr, n_classes)).to(dev)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
    else:
        weight_t = None
        criterion = nn.CrossEntropyLoss()
    adj = logit_adjust.to(dev) if logit_adjust is not None else None

    x_tr_t = torch.from_numpy(x_tr_s).to(dev)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
    x_va_t = torch.from_numpy(x_va_s).to(dev)
    x_test_t = torch.from_numpy(x_test_s).to(dev)

    best_score = -np.inf
    best_val_pred = np.zeros((len(x_va_s), n_classes))
    best_test_pred = np.zeros((len(x_test_s), n_classes))
    patience_counter = 0
    n = len(x_tr_t)

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(x_tr_t[idx])
            if adj is not None:
                # Adjusted logits in the loss; raw logits at inference.
                logits = logits + adj
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_proba = torch.softmax(model(x_va_t), dim=1).cpu().numpy()
        epoch_score = float(score(y_va, val_proba))
        if epoch_score > best_score:
            best_score = epoch_score
            best_val_pred = val_proba
            with torch.no_grad():
                best_test_pred = torch.softmax(model(x_test_t), dim=1).cpu().numpy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    del model, optimizer, scheduler, criterion, weight_t, adj
    del x_tr_t, y_tr_t, x_va_t, x_test_t
    return best_val_pred, best_test_pred
