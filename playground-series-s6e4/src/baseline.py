import pandas as pd

# TODO: set these after inspecting the downloaded data.
TARGET_COL = "target"
TASK = "classification"  # "classification" or "regression"

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

if TASK == "classification":
    prediction = train_df[TARGET_COL].mode().iloc[0]
else:
    prediction = train_df[TARGET_COL].mean()

submission = pd.DataFrame({"id": test_df["id"], TARGET_COL: prediction})
submission.to_csv("../data/baseline_submission.csv", index=False)
print("Baseline submission saved to data/baseline_submission.csv")
