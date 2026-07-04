"""From-scratch RealMLP-TD classifier (pure PyTorch + sklearn).

A faithful port of cdeotte's RealMLP re-implementation (via s6e6's
``realmlp_deotte.py``, notebook OOF 0.96881). This is NOT pytabkit — the
whole point of the re-implementation is the loss knob pytabkit lacks:
``loss_prior_power`` (balanced-softmax / logit adjustment), the lever that
fixed rare-class recall where class weights and oversampling were flat.

Architecture: PBLD periodic numeric embeddings, per-feature categorical
embeddings/one-hot, an internal ensemble of ``n_ens`` parallel MLPs (NTP
linears with per-group learning rates), front scaling layer, EMA weights,
flat-cos LR schedule, label smoothing.

The estimator is competition-agnostic: it takes a DataFrame plus
``cat_col_names`` and infers ``n_classes`` from the labels. Tune via
constructor kwargs over :data:`REALMLP_TD_CONFIG`.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 — torch convention
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from data_science_stuff.kaggle.models.losses import smooth_ce_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """Median-center / robust-scale / smooth-clip / l2-normalize pipeline."""

    def __init__(self, tfms: Sequence[str]) -> None:
        self._tfms = [
            t
            for t in tfms
            if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")
        ]

    def fit(
        self, X: NDArray[Any], y: NDArray[Any] | None = None
    ) -> NumericalPreprocessor:
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[np.float32]:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return np.asarray(X, dtype=np.float32)


class CategoricalFeatureLayer(nn.Module):
    """One-hot small-cardinality cats; per-ensemble-member embeddings for the rest."""

    def __init__(
        self,
        n_ens: int,
        cat_dims: Sequence[int],
        embed_dim: int = 8,
        onehot_thresh: int = 8,
    ) -> None:
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = list(cat_dims)
        self.onehot_features: list[int] = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices: list[int] = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList(
                    [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]
                )
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ens, _ = x.shape
        features: list[torch.Tensor] = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx : idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_module, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            emb_list = cast("nn.ModuleList", emb_module)
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx : feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            features.append(torch.cat(feat_embs, dim=1))
        if not features:  # no categorical columns at all
            return x.new_zeros(batch_size, n_ens, 0)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    """Learned per-feature input scale (one per ensemble member)."""

    def __init__(self, n_ens: int, n_features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    """Neural-tangent-parameterized linear layer over the internal ensemble."""

    def __init__(
        self, n_ens: int, in_features: int, out_features: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay embedding for numerical features."""

    def __init__(
        self,
        n_ens: int,
        n_features: int,
        hidden_dim: int = 16,
        out_dim: int = 4,
        freq_scale: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1)
            / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        periodic = torch.cos(
            2
            * math.pi
            * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0)
        )
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    """The RealMLP body: embeddings → scaled MLP ensemble → softmax probs."""

    def __init__(
        self,
        output_dim: int,
        cat_dims: Sequence[int],
        n_numerical: int,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens,
            cat_dims=cat_dims,
            embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(
            c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims
        )
        total_dim = num_emb_dim + cat_emb_dim
        act = cfg["activation"]
        layers: list[nn.Module] = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules: list[nn.Dropout] = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(cfg["hidden_dims"]):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(
            n_ens=n_ens, in_features=in_dim, out_features=output_dim
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)


def apply_schedule(
    init_value: float, progress: float, sched: str, flat_ratio: float = 0.3
) -> float:
    """Scalar schedules used for LR / label-smoothing / dropout over training."""
    if sched == "constant":
        return init_value
    if sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    if sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    if sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    msg = f"Unknown schedule: '{sched}'"
    raise ValueError(msg)


def get_parameter_groups(model: RealMLP, p: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-group LR/WD multipliers: scale, PBLD, first layer, weights, biases."""
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    lr, wd = p["lr"], p["weight_decay"]
    return [
        {
            "params": scale_p,
            "lr": lr * p["lr_scale_mult"],
            "weight_decay": wd * p["wd_scale_mult"],
            "group": "scale",
        },
        {
            "params": pbld_p,
            "lr": lr * p["pbld_lr_factor"],
            "weight_decay": wd,
            "group": "pbld",
        },
        {
            "params": first_w_p,
            "lr": lr * p["first_layer_lr_factor"],
            "weight_decay": wd * p["first_layer_wd_factor"],
            "group": "first_w",
        },
        {"params": other_w_p, "lr": lr, "weight_decay": wd, "group": "other_w"},
        {
            "params": bias_p,
            "lr": lr * p["lr_bias_mult"],
            "weight_decay": wd * p["wd_bias_mult"],
            "group": "bias",
        },
    ]


class RealMLP_TD_Classifier(BaseEstimator):
    """sklearn-style RealMLP-TD with the ``loss_prior_power`` imbalance knob.

    Construct with keyword overrides of :data:`REALMLP_TD_CONFIG`, e.g.
    ``RealMLP_TD_Classifier(epochs=8, loss_prior_power=1.075)``. ``fit``
    takes explicit train/val frames because early stopping and the EMA
    snapshot are validation-driven; call it once per CV fold.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params = {**REALMLP_TD_CONFIG, **kwargs}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: NDArray[Any],
        X_val: pd.DataFrame,
        y_val: NDArray[Any],
        cat_col_names: Sequence[str] | None = None,
        ckpt_path: str | Path | None = None,
        X_test: pd.DataFrame | None = None,
        per_sample_weight: NDArray[Any] | None = None,
    ) -> RealMLP_TD_Classifier:
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = list(cat_col_names or [])
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]
        if ckpt_path is None:
            ckpt_path = Path(tempfile.gettempdir()) / f"realmlp_ckpt_{os.getpid()}.pth"
        ckpt_path = Path(ckpt_path)

        X_tr_num: NDArray[np.float32] = (
            X_train[num_col_names].to_numpy().astype(np.float32)
        )
        X_val_num: NDArray[np.float32] = (
            X_val[num_col_names].to_numpy().astype(np.float32)
        )
        X_tr_cat: NDArray[np.int64] = X_train[cat_col_names].to_numpy().astype(np.int64)
        X_val_cat: NDArray[np.int64] = X_val[cat_col_names].to_numpy().astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].to_numpy().astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims
        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_tr
        )
        cw_power = float(p.get("class_weight_power", 1.0))
        if cw_power != 1.0:
            weights_np = np.power(weights_np, cw_power)
        cw_mult = p.get("class_weight_multipliers", None)
        if cw_mult is not None:
            cw_mult = np.asarray(cw_mult, dtype="float64")
            weights_np = weights_np * cw_mult
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        loss_prior_power = float(p.get("loss_prior_power", 0.0))
        loss_prob_multipliers = None
        if loss_prior_power != 0.0:
            class_counts = np.bincount(y_tr, minlength=len(classes)).astype("float64")
            class_counts = class_counts / np.exp(np.log(class_counts).mean())
            loss_prob_multipliers = torch.as_tensor(
                np.power(class_counts, loss_prior_power),
                dtype=torch.float32,
                device=dev,
            )

        n_classes = len(classes)
        base_model = RealMLP(
            output_dim=n_classes,
            cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1],
            cfg=p,
        ).to(dev)
        self.model_ = base_model  # eager handle for eval / state_dict / predict
        param_groups = get_parameter_groups(base_model, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        # Speed knobs (all opt-in; default-off keeps the stock run identical):
        #   fused_optimizer → fused AdamW kernel; allow_amp → fp16 autocast
        #   (softmax/log auto-promote to fp32); compile_model → Inductor fwd.
        use_fused = bool(p.get("fused_optimizer", False)) and dev.type == "cuda"
        optimizer = torch.optim.AdamW(
            param_groups, betas=(p["mom"], p["sq_mom"]), fused=use_fused
        )
        amp_device = "cuda" if dev.type == "cuda" else "cpu"
        use_amp = bool(p.get("allow_amp", False)) and dev.type == "cuda"
        scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
        forward_model: Any = base_model
        if bool(p.get("compile_model", False)):  # pragma: no cover — Inductor JIT
            forward_model = torch.compile(base_model, dynamic=True)

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        per_sample_w_t = None
        if per_sample_weight is not None:
            per_sample_w_t = torch.as_tensor(
                np.asarray(per_sample_weight, dtype=np.float32),
                dtype=torch.float32,
                device=dev,
            )
            if per_sample_w_t.shape[0] != len(y_tr):
                msg = (
                    f"per_sample_weight length {per_sample_w_t.shape[0]} "
                    f"!= n_train {len(y_tr)}"
                )
                raise ValueError(msg)

        n_ens = p["n_ens"]
        train_bs, eval_bs, epochs = p["train_bs"], p["eval_bs"], p["epochs"]
        lr_sched, flat_ratio = p["lr_sched"], p["flat_ratio"]
        numeric_noise_std = float(p.get("numeric_noise_std", 0.0))
        ema_decay = float(p.get("ema_decay", 0.0))
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))
        sample_weight_power = float(p.get("sample_weight_power", 0.0))
        sampler_rng = np.random.default_rng(int(p.get("random_state", 0)) + 2027)
        sample_probs = None
        if sample_weight_power > 0:
            class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
            sample_probs = np.power(
                1.0 / np.clip(class_counts[y_tr], 1.0, None), sample_weight_power
            )
            sample_probs = sample_probs / sample_probs.sum()

        best_score, best_epoch = -np.inf, 0
        best_val_probs: NDArray[Any] | None = None
        self.ckpt_path_ = ckpt_path
        ema_state = None
        if ema_decay > 0:
            ema_state = {
                k: v.detach().clone() for k, v in self.model_.state_dict().items()
            }

        for epoch in range(epochs):
            self.model_.train()
            if sample_probs is not None:
                epoch_order = sampler_rng.choice(
                    len(y_tr), size=len(y_tr), replace=True, p=sample_probs
                )
            else:
                epoch_order = train_order
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = epoch_order[start : start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(
                        g["lr_base"], progress, lr_sched, flat_ratio
                    )
                optimizer.zero_grad()
                x_num_batch = Xtn[idx_batch]
                if numeric_noise_std > 0:
                    x_num_batch = (
                        x_num_batch + torch.randn_like(x_num_batch) * numeric_noise_std
                    )
                ls_val = apply_schedule(
                    p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio
                )
                drop_val = apply_schedule(
                    p["dropout"], progress, p["p_drop_sched"], flat_ratio
                )
                for dm in base_model._dropout_modules:
                    dm.p = drop_val
                psw_batch = None
                if per_sample_w_t is not None:
                    psw_batch = per_sample_w_t[idx_batch].repeat_interleave(n_ens)
                with torch.autocast(amp_device, dtype=torch.float16, enabled=use_amp):
                    y_pred = forward_model(x_num_batch, Xtc[idx_batch])
                    loss = smooth_ce_loss(
                        ytt[idx_batch].repeat_interleave(n_ens),
                        y_pred.reshape(-1, n_classes),
                        ls=ls_val,
                        class_weights=class_weights,
                        focal_gamma=float(p.get("focal_gamma", 0.0)),
                        loss_prob_multipliers=loss_prob_multipliers,
                        per_sample_weight=psw_batch,
                    )
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), p["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if ema_state is not None:
                    with torch.no_grad():
                        for key, value in self.model_.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(
                                    value.detach(), alpha=1.0 - ema_decay
                                )
                            else:
                                ema_state[key].copy_(value)
            if sample_probs is None:
                np.random.shuffle(train_order)

            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate(
                    [
                        self.model_(Xvn[s : s + eval_bs], Xvc[s : s + eval_bs])
                        .mean(dim=1)
                        .cpu()
                        .numpy()
                        for s in range(0, len(y_v), eval_bs)
                    ],
                    axis=0,
                )
            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)

            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            improved = epoch_score > best_score
            if improved:
                best_score, best_epoch = epoch_score, epoch + 1
                best_val_probs = val_probs.copy()
                torch.save(
                    ema_state if ema_state is not None else self.model_.state_dict(),
                    ckpt_path,
                )
            if verbose >= 2:
                print(
                    f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  "
                    f"best={best_score:.5f}" + (" ✓" if improved else "")
                )
            if p["use_early_stopping"]:
                patience = (
                    best_epoch * p["early_stopping_multiplicative_patience"]
                    + p["early_stopping_additive_patience"]
                )
                if (epoch + 1) > patience:
                    break

        self.model_.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  → best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[Any]:
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(
            X[self.num_col_names_].to_numpy().astype(np.float32)
        )
        X_cat: NDArray[np.int64] = X[self.cat_col_names_].to_numpy().astype(np.int64)
        if self.cat_dims_:
            X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            proba: NDArray[Any] = np.concatenate(
                [
                    self.model_(Xn[s : s + eval_bs], Xc[s : s + eval_bs])
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                    for s in range(0, len(X_num), eval_bs)
                ],
                axis=0,
            )
        return proba

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        labels: NDArray[Any] = self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return labels


REALMLP_TD_CONFIG: dict[str, Any] = {
    "n_ens": 8,
    "embed_dim": 7,
    "onehot_thresh": 10,
    "hidden_dims": [512, 512, 512],
    "dropout": 0.041,
    "p_drop_sched": "expm4t",
    "activation": nn.GELU,
    "add_front_scale": True,
    "pbld_hidden_dim": 16,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 2.33,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.115,
    "lr": 0.01,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.175,
    "first_layer_lr_factor": 0.5,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.013,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    "class_weight_power": 0.0,
    "class_weight_multipliers": None,
    "sample_weight_power": 0.0,
    "loss_prior_power": 1.075,
    "focal_gamma": 0.0,
    "eval_class_multipliers": None,
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "epochs": 6,
    "train_bs": 256,
    "eval_bs": 10240,
    "numeric_noise_std": 0.0,
    "ema_decay": 0.99775,
    "verbosity": 1,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "device": str(DEVICE),
    "random_state": 42,
}
