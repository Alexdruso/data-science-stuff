# Kaggle Playground Series S6E6 — Predicting Stellar Class

**Goal**: Predict stellar class: GALAXY, STAR, or QSO.
**Evaluation**: Balanced accuracy.
**Deadline**: June 30, 2026.

Dataset is synthetically generated from real stellar observation data (SDSS).

## Project Structure

- `data/` — Train/test CSVs (download via Kaggle CLI, gitignored)
- `src/` — Python modules (`features.py` is the SSOT for all feature engineering)
- `notebooks/` — EDA scripts
- `submissions/` — Output prediction CSVs
- `results/` — CV scores, model artifacts, analysis outputs

## Setup

```bash
# From repo root
source .venv/bin/activate

# Download competition data
kaggle competitions download -c playground-series-s6e6 -p playground-series-s6e6/data/
cd playground-series-s6e6/data && unzip playground-series-s6e6.zip && rm playground-series-s6e6.zip && cd ../..
```

## Running

```bash
python playground-series-s6e6/src/baseline.py
python playground-series-s6e6/notebooks/eda.py
```

## Citation

Yao Yan, Walter Reade, Elizabeth Park. Predicting Stellar Class. https://kaggle.com/competitions/playground-series-s6e6, 2026. Kaggle.
