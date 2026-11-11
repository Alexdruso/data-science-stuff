"""Package src/lm_leg.py as a single-file Kaggle GPU kernel that runs several jobs in parallel.

    python src/kaggle_lm_build.py SLUG "ARGS FOR GPU0" ["ARGS FOR GPU1"] [--ds=owner/dataset] [--src=clm_leg.py]
    kaggle kernels push -p kaggle_kernels/SLUG --accelerator NvidiaTeslaT4

Each job is one `lm_leg.py` argument string run on its own GPU (CUDA_VISIBLE_DEVICES).
Outputs land in /kaggle/working/_<name>/f{k}_{val,test}.npy (+ model .pt deleted to save
space unless --keep_models); download them into results/_<name>/ and re-run lm_leg.py locally
to assemble (cached folds are skipped).
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SHIM = '''
# ---- Kaggle shims for the repo imports --------------------------------------------------
import os, subprocess
from sklearn.model_selection import StratifiedKFold
TARGET = "is_fraud"
NUM_COLS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
            "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
            "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score",
            "prior_disputes"]
CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
RAW_COLS = NUM_COLS + CAT_COLS + BOOL_COLS


def build_features(df):
    df = df.sort("id")
    return df.with_columns(pl.col(TARGET).cast(pl.Int8)) if TARGET in df.columns else df


def fold_assignment(y):
    folds = np.zeros(len(y), dtype=np.int64)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y), 1):
        folds[va] = f
    return folds


def _find(name):
    for p in Path("/kaggle/input").rglob(name):
        return p
    raise FileNotFoundError(name)


DATA_DIR = Path("/kaggle/working/data")
(DATA_DIR / "orig").mkdir(parents=True, exist_ok=True)
for _n in ("train.csv", "test.csv"):
    if not (DATA_DIR / _n).exists():
        os.symlink(_find(_n), DATA_DIR / _n)
if not (DATA_DIR / "orig" / "credit_card_fraud_2026.csv").exists():
    os.symlink(_find("credit_card_fraud_2026.csv"), DATA_DIR / "orig" / "credit_card_fraud_2026.csv")
RESULTS_DIR = Path("/kaggle/working")
for _p in Path("/kaggle/input").rglob("_lm_*_orig_e*.pt"):  # replica checkpoints (--ds)
    if not (RESULTS_DIR / _p.name).exists():
        os.symlink(_p, RESULTS_DIR / _p.name)
SUBMISSIONS_DIR = Path("/kaggle/working")
# ------------------------------------------------------------------------------------------
'''

DRIVER = '''

JOBS = {jobs!r}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        procs = []
        for g, args in enumerate(JOBS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log = open(f"/kaggle/working/job{{g}}.log", "w")
            procs.append(subprocess.Popen([sys.executable, __file__, *args.split()], env=env,
                                          stdout=log, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        for g in range(len(JOBS)):
            print(f"===== job {{g}}: {{JOBS[g]}}")
            print(open(f"/kaggle/working/job{{g}}.log").read()[-6000:])
        if not {keep}:
            for pt in Path("/kaggle/working").rglob("*_model.pt"):
                pt.unlink()
        for d in Path("/kaggle/working/data").rglob("*.csv"):
            d.unlink()
'''


def main() -> None:
    ds = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--ds=")]
    args = [x for x in sys.argv[1:] if not x.startswith(("--keep", "--ds=", "--src="))]
    srcs = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--src=")]
    keep = "--keep_models" in sys.argv
    slug, jobs = args[0], args[1:]
    src = (HERE / (srcs[0] if srcs else "lm_leg.py")).read_text()
    # drop the repo imports and path setup; the shim defines the same names
    src = re.sub(r"sys\.path\.insert.*\n", "", src)
    src = re.sub(r"from features import .*\n", "", src)
    src = re.sub(r"from data_science_stuff\.kaggle\.cv import .*\n", "", src)
    src = re.sub(r"from data_science_stuff\.kaggle\.io import competition_dirs\n", "", src)
    src = re.sub(r"DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR = competition_dirs\(__file__\)\n",
                 SHIM, src)
    src = src.replace("    if all((out / f", "    if False and all((out / f")  # assemble locally
    src = src.replace('\n\nif __name__ == "__main__":\n    main()\n', "")
    src += DRIVER.format(jobs=jobs, keep=keep)
    out = HERE.parent / "kaggle_kernels" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_job.py").write_text(src)
    meta = {
        "id": f"alessandrosanvito/{slug}", "title": slug, "code_file": "lm_job.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["uditjain13/credit-card-fraud-detection-2026", *ds],
        "competition_sources": ["playground-series-s6e11"], "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(out)


if __name__ == "__main__":
    main()
