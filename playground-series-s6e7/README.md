# Kaggle Playground Series S6E7 — Predicting Student Health Risk

**Goal**: Predict a student's `health_condition` (`at-risk`, `fit`, `unhealthy`) from
lifestyle/biometric features — 3-class classification.
**Evaluation**: Balanced accuracy (mean per-class recall).
**Deadline**: ~August 2, 2026.

## Project Structure

- `data/` — Train/test CSVs (download via Kaggle CLI, gitignored)
- `src/` — Python modules (`features.py`, `train_*.py`, `ensemble.py`, `eda.py`)
- `notebooks/` — EDA notebooks
- `submissions/` — Output prediction CSVs
- `results/` — OOF/test `.npy` arrays, `cv_scores.csv`, decision weights, `eda_summary.md`

## Setup

```bash
kaggle competitions download -c playground-series-s6e7 -p data/
cd data && unzip playground-series-s6e7.zip && rm playground-series-s6e7.zip && cd ..
```

## Running

```bash
python src/eda.py             # distributions + adversarial validation
python src/baseline.py        # LGBM 5-fold
python src/train_xgboost.py   # XGBoost (GPU)
python src/train_catboost.py  # CatBoost (GPU)
python src/ensemble.py        # two-stage blend → submissions/ensemble_v1.csv
```

See `CLAUDE.md` for dataset details, EDA findings, conventions, and the experiments log.
