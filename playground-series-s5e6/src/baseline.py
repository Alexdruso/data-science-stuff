import pandas as pd
import numpy as np

# Read the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Get the top 3 most common fertilizers
top_fertilizers = train_df['Fertilizer Name'].value_counts().nlargest(3).index.tolist()
print("Top 3 most common fertilizers:", top_fertilizers)

# Create submission
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['Fertilizer Name'] = ' '.join(top_fertilizers)

# Save submission
submission.to_csv('../data/baseline_submission.csv', index=False)
print("Baseline submission saved to data/baseline_submission.csv") 
