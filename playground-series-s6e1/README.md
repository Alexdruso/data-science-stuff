# Kaggle Playground Series S6E1 - Binary Classification with a Flood Prediction Dataset

This repository contains my solution for the Kaggle Playground Series S6E1 competition, which focuses on predicting whether it will rain tomorrow based on weather data.

## Competition Overview

- **Goal**: Predict whether it will rain tomorrow (binary classification)
- **Evaluation Metric**: Area Under the ROC Curve (AUC)
- **Timeline**: [Insert timeline if known]

## Project Structure

```
playground-series-s6e1/
├── data/               # Data files
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
└── README.md         # This file
```

## Setup Instructions

1. Download the competition data:
```bash
kaggle competitions download -c playground-series-s6e1
```

2. Move the downloaded files to the `data/` directory.

## Evaluation Metric

The competition uses Area Under the ROC Curve (AUC) as the evaluation metric.

## Submission Format

The submission file should contain:
- A header row
- Two columns: `id` and `rainfall`

Example:
```
id,rainfall
750000,0
```

## Citation

[Insert citation if available]

