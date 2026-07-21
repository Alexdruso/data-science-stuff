"""RealMLP-TD config ladder on one fixed fold (Day-12 GPU track).

RealMLP has NEVER been tuned on this dataset — the 0.9506 flagship runs
pytabkit-TD meta-learned defaults, and `epochs: 6` (~6.5k steps at bs 512) on
a 552k-row fold is the single most suspicious knob. At ~75-80 min per full
5-fold config, TPE is the wrong tool: this is a PRE-REGISTERED ladder of ~12
single-axis configs screened on seed-42 fold 1 (the exact zoo_cv split, te13
recipe, unrepaired surface — all TE caches hit), scored by fold-1 argmax AND
decision-weighted bacc on the 138k val rows. Config 0 = the current trainer
CFG (paired baseline; fold-1 argmax reference ~0.9511).

Already measured, NOT re-swept: epoch_select="bll" (negative), snapshot_k
(wash), the kNN block (flat).

Stage A sweeps epochs {10,14,20}; stage B runs lr {0.006,0.02} at the best
stage-A epochs; the remaining axes are single-axis deviations from the
default config so every read is paired vs config 0.

Resumable: per-config results checkpoint results/tune_realmlp_td.json —
a reboot mid-ladder skips finished configs (this box hard-reboots at random).
GPU memory rule: del model + empty_cache per config; OOM retries once at
train_bs=256, else the config is recorded as OOM and skipped.

Promotion: fold-1 dw delta >= +0.0007 over config 0 -> confirm through the
full trainer (S6E7_REALMLP_PARAMS='<json>' S6E7_TE13=1 S6E7_SEEDS=42
S6E7_RUN_TAG=_u_s42t python src/train_realmlp_td.py); PASS = weighted OOF
>= 0.9515 (incumbent 0.9505 + 0.001).

Run:  ../.venv/bin/python src/tune_realmlp_td.py | tee -a results/tune_realmlp_td.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_common import N_CLASSES, N_FOLDS, RESULTS_DIR, load_dataset
from train_mlp import NNData
from zoo_common import te13_block_for_fold, te_block_for_fold

from data_science_stuff.kaggle_utils import tune_decision_weights, weighted_predict

SEED = 42
FOLD = 1  # 1-indexed zoo_cv fold; fold_seed matches the trainer exactly
FOLD_SEED = SEED + FOLD * 100
CKPT = RESULTS_DIR / "tune_realmlp_td.json"

# The trainer's CFG (train_realmlp_td.py) = config 0. Every ladder entry is an
# override dict on top of this.
BASE_CFG: dict[str, object] = dict(
    train_bs=512,
    fused_optimizer=True,
    allow_amp=True,
    verbosity=1,
    epoch_select="bacc",
    snapshot_k=0,
)

STAGE_A: list[tuple[str, dict[str, object]]] = [
    ("base", {}),
    ("ep10", {"epochs": 10}),
    ("ep14", {"epochs": 14}),
    ("ep20", {"epochs": 20}),
]

# Single-axis deviations from the default config (paired vs base).
STAGE_C: list[tuple[str, dict[str, object]]] = [
    ("lpp1.0", {"loss_prior_power": 1.0}),
    ("lpp1.2", {"loss_prior_power": 1.2}),
    ("wd0.04", {"weight_decay": 0.04}),
    ("do0.10", {"dropout": 0.10}),
    ("hid768", {"hidden_dims": [768, 768, 768]}),
    ("ls0", {"ls_eps": 0.0}),
    ("noise0.02", {"numeric_noise_std": 0.02}),
]

# Day-14 training-dynamics stage: manifold mixup (vicinal risk — the one
# augmentation mechanism with zero campaign coverage) and SAM (the Day-8
# asserted-kill the Day-9 persona notes re-armed; fp32 by construction).
STAGE_D: list[tuple[str, dict[str, object]]] = [
    ("mix0.2", {"mixup_alpha": 0.2}),
    ("mix1.0", {"mixup_alpha": 1.0}),
    ("sam0.05", {"sam_rho": 0.05, "allow_amp": False}),
    ("sam0.10", {"sam_rho": 0.10, "allow_amp": False}),
]

# Day-17 S2 stage: soft-confusion-matrix balanced-accuracy surrogate (ideas_day11
# S2, armed Day-16) — the last unmeasured objective slot. The surrogate REPLACES
# the balanced-softmax mechanism (loss_prior_power -> 0), pure and 50/50-CE arms.
STAGE_S2: list[tuple[str, dict[str, object]]] = [
    ("softba1.0", {"soft_ba_weight": 1.0, "loss_prior_power": 0.0}),
    ("softba0.5", {"soft_ba_weight": 0.5, "loss_prior_power": 0.0}),
]


def load_ckpt() -> dict[str, dict[str, object]]:
    if CKPT.exists():
        with CKPT.open() as f:
            return json.load(f)
    return {}


def save_ckpt(results: dict[str, dict[str, object]]) -> None:
    with CKPT.open("w") as f:
        json.dump(results, f, indent=2)


def build_fold1_design():  # noqa: ANN201
    """The exact train_realmlp_td.fit_fold input pipeline for fold 1 (te13 on,
    u surface), built ONCE and shared by every ladder config."""
    ds = load_dataset()
    nd = NNData(ds)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = dict(enumerate(skf.split(ds.y, ds.y), 1))
    tr_idx, val_idx = splits[FOLD]

    te_tr_full, _te_te, _n = te_block_for_fold(ds, tr_idx, SEED, FOLD, "u")
    t13_tr, _t13_te, _n13 = te13_block_for_fold(ds, tr_idx, SEED, FOLD, "u")
    te_tr_full = np.concatenate([te_tr_full, t13_tr], axis=1)

    num_med = np.nanmedian(nd.Xnum_tr[tr_idx], axis=0)
    num_tr = np.where(np.isnan(nd.Xnum_tr), num_med, nd.Xnum_tr)
    te_med = np.nanmedian(te_tr_full[tr_idx], axis=0)
    te_tr_f = np.where(np.isnan(te_tr_full), te_med, te_tr_full)

    cat_names = [f"cat_{c}" for c in nd.cat_cols]
    blocks = {
        **{n: nd.Xcat_tr[:, i] for i, n in enumerate(cat_names)},
        **{n: num_tr[:, i] for i, n in enumerate(nd.num_cols)},
        **{f"miss_{i}": nd.miss_tr[:, i] for i in range(nd.miss_tr.shape[1])},
        **{f"te_{i}": te_tr_f[:, i] for i in range(te_tr_f.shape[1])},
    }
    X_all = pd.DataFrame(blocks)
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    print(
        f"fold {FOLD} design: {X_tr.shape[0]} train / {X_val.shape[0]} val, "
        f"{X_all.shape[1]} cols ({te_tr_f.shape[1]} TE)"
    )
    return X_tr, X_val, ds.y[tr_idx], ds.y[val_idx], cat_names


def run_config(name, overrides, X_tr, X_val, y_tr, y_val, cat_names):  # noqa: ANN001, ANN201
    from data_science_stuff.kaggle.models.realmlp import RealMLP_TD_Classifier

    for attempt_bs in (overrides.get("train_bs", BASE_CFG["train_bs"]), 256):
        cfg = {**BASE_CFG, **overrides, "train_bs": attempt_bs, "random_state": FOLD_SEED}
        torch.manual_seed(FOLD_SEED)
        np.random.seed(FOLD_SEED)  # noqa: NPY002 — library-internal RNG expects it
        t0 = time.time()
        try:
            model = RealMLP_TD_Classifier(**cfg)
            model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_names)
            assert list(model.classes_) == list(range(N_CLASSES)), model.classes_
            proba = model.best_val_probs_.astype(np.float64)
            proba /= proba.sum(axis=1, keepdims=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            argmax = float(balanced_accuracy_score(y_val, proba.argmax(axis=1)))
            w = tune_decision_weights(y_val, proba, n_restarts=3)
            dw = float(balanced_accuracy_score(y_val, weighted_predict(proba, w)))
            return {
                "argmax": argmax,
                "dw": dw,
                "overrides": overrides,
                "train_bs": attempt_bs,
                "seconds": round(time.time() - t0, 1),
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt_bs == 256:
                break
            print(f"  [{name}] OOM at train_bs={attempt_bs}, retrying at 256")
    return {"oom": True, "overrides": overrides}


def main() -> None:
    design = build_fold1_design()
    results = load_ckpt()

    def run_stage(stage: list[tuple[str, dict[str, object]]]) -> None:
        for name, overrides in stage:
            if name in results:
                print(f"[{name}] cached: {results[name]}")
                continue
            print(f"[{name}] overrides={overrides}")
            results[name] = run_config(name, overrides, *design)
            save_ckpt(results)
            print(f"[{name}] -> {results[name]}")

    run_stage(STAGE_A)

    # Stage B: lr sweep at the best stage-A epochs (deterministic from ckpt).
    scored_a = {n: r for n, r in results.items() if "dw" in r and n in dict(STAGE_A)}
    best_a = max(scored_a, key=lambda n: scored_a[n]["dw"])
    best_epochs = dict(STAGE_A)[best_a].get("epochs", 6)
    print(f"stage A best: {best_a} (dw={scored_a[best_a]['dw']:.5f}) "
          f"-> stage B at epochs={best_epochs}")
    stage_b = [
        (f"lr0.006_ep{best_epochs}", {"lr": 0.006, "epochs": best_epochs}),
        (f"lr0.02_ep{best_epochs}", {"lr": 0.02, "epochs": best_epochs}),
    ]
    run_stage(stage_b)
    run_stage(STAGE_C)
    run_stage(STAGE_D)
    run_stage(STAGE_S2)

    base_dw = results["base"]["dw"]
    base_am = results["base"]["argmax"]
    print(f"\n=== ladder results (fold {FOLD}, base dw={base_dw:.5f} "
          f"argmax={base_am:.5f}) ===")
    scored = {n: r for n, r in results.items() if "dw" in r}
    promoted = []
    for name, r in sorted(scored.items(), key=lambda kv: -kv[1]["dw"]):
        d_dw = r["dw"] - base_dw
        d_am = r["argmax"] - base_am
        mark = ""
        if name != "base" and d_dw >= 0.0007:
            promoted.append(name)
            mark = "  << PROMOTE"
        print(
            f"{name:>16}  dw={r['dw']:.5f} ({d_dw:+.5f})  "
            f"argmax={r['argmax']:.5f} ({d_am:+.5f})  {r['seconds']:.0f}s{mark}"
        )
    for name, r in results.items():
        if r.get("oom"):
            print(f"{name:>16}  OOM (both batch sizes)")
    if promoted:
        print(
            f"\nPROMOTE {promoted}: confirm via "
            f"S6E7_REALMLP_PARAMS='<overrides json>' S6E7_TE13=1 S6E7_SEEDS=42 "
            f"S6E7_RUN_TAG=_u_s42t python src/train_realmlp_td.py "
            f"(gate: weighted OOF >= 0.9515)"
        )
    else:
        print(
            "\nFAIL: no config clears +0.0007 on fold 1 — "
            "RealMLP HP axis closed by measurement."
        )


if __name__ == "__main__":
    main()
