# Kaggle Playground Series S6E5 — Predicting F1 Pit Stops

**Goal**: Predict whether a Formula 1 driver will pit on the next lap (`PitNextLap`, binary).  
**Evaluation**: Area under the ROC curve (AUC-ROC).  
**Deadline**: May 31, 2026.

Dataset is synthetically generated from a real F1 strategy dataset. `Normalized_TyreLife` is intentionally excluded to avoid trivial prediction.

## Project Structure

- `data/` — Train/test CSVs (download via Kaggle CLI)
- `src/` — Python modules
- `notebooks/` — EDA notebooks
- `submissions/` — Output prediction CSVs
- `results/` — CV scores and analysis outputs

## Setup

```bash
# From competition directory
kaggle competitions download -c playground-series-s6e5 -p data/
cd data && unzip playground-series-s6e5.zip && rm playground-series-s6e5.zip && cd ..
```

## Running

```bash
python src/baseline.py
```

## Citation

Walter Reade and Elizabeth Park. Predicting F1 Pit Stops. https://kaggle.com/competitions/playground-series-s6e5, 2026. Kaggle.
