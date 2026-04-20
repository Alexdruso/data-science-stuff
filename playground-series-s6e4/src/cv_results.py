"""Shared utility for appending CV scores to results/cv_scores.csv."""

from datetime import datetime
from pathlib import Path

import pandas as pd


def save_cv_result(
    results_dir: Path,
    model: str,
    fold_accs: list[float],
    oof_acc: float,
) -> None:
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "cv_scores.csv"

    row: dict[str, object] = {
        "model": model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oof_accuracy": round(oof_acc, 6),
    }
    for i, acc in enumerate(fold_accs, 1):
        row[f"fold_{i}_acc"] = round(acc, 6)

    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        pd.concat([existing, new_row], ignore_index=True).to_csv(path, index=False)
    else:
        new_row.to_csv(path, index=False)

    print(f"CV result appended → {path}")
