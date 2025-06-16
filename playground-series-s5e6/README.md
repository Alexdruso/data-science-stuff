# Kaggle Playground Series S5E6 - Fertilizer Prediction

This repository contains my solution for the Kaggle Playground Series S5E6 competition, which focuses on predicting the optimal fertilizer for different weather, soil conditions, and crops.

## Competition Overview

- **Goal**: Select the best fertilizer for different weather, soil conditions, and crops
- **Evaluation Metric**: Mean Average Precision @ 3 (MAP@3)
- **Timeline**: June 1, 2025 - June 30, 2025

## Project Structure

```
playground-series-s5e6/
├── data/               # Data files
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
└── README.md         # This file
```

## Setup Instructions

1. Download the competition data:
```bash
kaggle competitions download -c playground-series-s5e6
```

2. Move the downloaded files to the `data/` directory.

## Evaluation Metric

The competition uses Mean Average Precision @ 3 (MAP@3) as the evaluation metric. For each observation, we need to predict up to 3 fertilizer names, and the predictions are evaluated based on their precision at each position.

## Submission Format

The submission file should contain:
- A header row
- Two columns: `id` and `Fertilizer Name`
- Up to 3 space-delimited fertilizer names per prediction

Example:
```
id,Fertilizer Name
750000,14-35-14 10-26-26 Urea
```

## Citation

Walter Reade and Elizabeth Park. Predicting Optimal Fertilizers. https://kaggle.com/competitions/playground-series-s5e6, 2025. Kaggle. 
