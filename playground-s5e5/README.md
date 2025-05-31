# Playground Series S5E5 - Initial Solution

This repository contains my solution for the Kaggle Playground Series S5E5 competition.

## Project Structure

```
playground-s5e5/
├── data/               # Competition data
├── src/               # Source code
├── notebooks/         # Jupyter notebooks for exploration
├── models/           # Saved models
└── submissions/      # Submission files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Initial Solution

The initial solution uses PyCaret for automated machine learning and Polars for efficient data processing. The approach includes:

1. Data preprocessing with Polars
2. Automated model selection and training with PyCaret
3. Feature engineering and selection
4. Outlier detection and removal
5. Model comparison and selection based on RMSE

To run the initial solution:
```bash
python src/baseline_solution.py
```

## Next Steps

1. Perform detailed exploratory data analysis
2. Implement custom feature engineering
3. Try different model architectures
4. Optimize hyperparameters
5. Implement cross-validation strategies 
