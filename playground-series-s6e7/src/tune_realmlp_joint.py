"""RealMLP joint HP search (Day-17, armed Day-16): red-team of the Day-12 ladder.

The Day-12 closure ("all 13 configs within [-0.0003,+0.0002]") is a SINGLE-AXIS
result: every config was one knob off the incumbent. A joint optimum the ladder
cannot see (e.g. more epochs x lower lr x more regularization) is the one
uncovered configuration channel. This runs TPE over 8 axes jointly on the exact
fold-1 screen instrument the ladder used (tune_realmlp_td.build_fold1_design +
run_config: seed-42 fold 1, te13 recipe, u surface, all TE caches hit), scored
by decision-weighted bacc on the 138k val rows.

Incumbent (trainer defaults) is enqueued as trial 0 — every read is paired.
Promotion bar: fold-1 dw delta >= +0.0007 over the incumbent (the ladder's
screen bar) -> confirm through the full trainer via S6E7_REALMLP_PARAMS.

Resumable: optuna sqlite storage results/tune_realmlp_joint.db — a reboot
mid-search resumes (this box hard-reboots at random). GPU memory rule lives in
run_config (del model + empty_cache per trial, OOM retry at bs 256).

Run:  ../.venv/bin/python src/tune_realmlp_joint.py --hours 3.5 \
          | tee -a results/tune_realmlp_joint.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).parent))
from train_common import RESULTS_DIR
from tune_realmlp_td import build_fold1_design, run_config

STUDY_NAME = "realmlp_joint_day17"
STORAGE = f"sqlite:///{RESULTS_DIR / 'tune_realmlp_joint.db'}"

# Trainer-default values for every swept axis (REALMLP_TD_CONFIG + BASE_CFG):
# the incumbent trial and the center of the search space.
INCUMBENT: dict[str, object] = {
    "lr": 0.01,
    "epochs": 6,
    "weight_decay": 0.013,
    "dropout": 0.041,
    "hidden": 512,
    "loss_prior_power": 1.075,
    "ls_eps": 0.04,
    "numeric_noise_std": 0.0,
}


def suggest(trial: optuna.Trial) -> dict[str, object]:
    return {
        "lr": trial.suggest_float("lr", 0.004, 0.03, log=True),
        "epochs": trial.suggest_int("epochs", 4, 16),
        "weight_decay": trial.suggest_float("weight_decay", 0.002, 0.08, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
        "hidden": trial.suggest_categorical("hidden", [256, 512, 768]),
        "loss_prior_power": trial.suggest_float("loss_prior_power", 0.9, 1.25),
        "ls_eps": trial.suggest_float("ls_eps", 0.0, 0.12),
        "numeric_noise_std": trial.suggest_float("numeric_noise_std", 0.0, 0.04),
    }


def to_overrides(params: dict[str, object]) -> dict[str, object]:
    o = dict(params)
    h = int(o.pop("hidden"))  # type: ignore[arg-type]
    o["hidden_dims"] = [h, h, h]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--trials", type=int, default=150)
    args = ap.parse_args()

    design = build_fold1_design()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=15, multivariate=True, group=True
        ),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(INCUMBENT)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        res = run_config(f"joint_{trial.number}", to_overrides(params), *design)
        if res.get("oom"):
            raise optuna.TrialPruned("OOM at both batch sizes")
        print(
            f"[trial {trial.number}] dw={res['dw']:.5f} argmax={res['argmax']:.5f} "
            f"{res['seconds']:.0f}s {params}",
            flush=True,
        )
        trial.set_user_attr("argmax", res["argmax"])
        trial.set_user_attr("seconds", res["seconds"])
        return float(res["dw"])

    study.optimize(objective, n_trials=args.trials, timeout=args.hours * 3600)

    done = [t for t in study.trials if t.value is not None]
    incumbent = next(
        (t for t in done if all(t.params.get(k) == v for k, v in INCUMBENT.items())),
        None,
    )
    base = incumbent.value if incumbent is not None else None
    best = study.best_trial
    print(f"\n=== joint TPE summary ({len(done)} scored trials) ===")
    if base is not None:
        print(f"incumbent dw={base:.5f}")
    print(f"best trial {best.number}: dw={best.value:.5f} params={best.params}")
    if base is not None and best.value is not None:
        delta = best.value - base
        print(f"delta vs incumbent: {delta:+.5f} (screen bar +0.0007)")
        if delta >= 0.0007:
            print(
                "PROMOTE: confirm via S6E7_REALMLP_PARAMS="
                f"'{to_overrides(best.params)}' S6E7_TE13=1 S6E7_SEEDS=42 "
                "S6E7_RUN_TAG=_u_s42j python src/train_realmlp_td.py"
            )
        else:
            print(
                "NULL: no joint config clears +0.0007 on fold 1 — the Day-12 "
                "single-axis closure survives its joint red-team."
            )


if __name__ == "__main__":
    main()
