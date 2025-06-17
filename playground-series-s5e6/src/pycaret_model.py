import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, finalize_model, predict_model, pull
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import label_ranking_average_precision_score

# Define custom MAP@3 metric
def map_at_3(y_true, y_pred_proba):
    # Convert true labels to one-hot encoding
    le = LabelEncoder()
    le.fit(y_true)
    y_true_encoded = le.transform(y_true)
    
    # Get top 3 predictions for each sample
    top_3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:]
    y_pred_top3 = np.zeros_like(y_pred_proba)
    for i, indices in enumerate(top_3_indices):
        y_pred_top3[i, indices] = 1
    
    # Calculate MAP@3
    return label_ranking_average_precision_score(y_true_encoded, y_pred_proba)

# Read the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Initialize PyCaret setup
clf = setup(
    data=train_df,
    target='Fertilizer Name',
    numeric_features=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'],
    categorical_features=['Soil Type', 'Crop Type'],
    verbose=False,
)

# Compare different models using MAP@3
best_model = compare_models(sort='Accuracy', n_select=1)

# Create final model
final_model = finalize_model(best_model)

# Make predictions on test data
predictions = predict_model(final_model, data=test_df)

# Get top 3 predictions for each sample
pred_proba = predict_model(final_model, data=test_df, raw_score=True)
top_3_indices = np.argsort(pred_proba, axis=1)[:, -3:]
le = LabelEncoder()
le.fit(train_df['Fertilizer Name'])

# Create submission with top 3 predictions
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['Fertilizer Name'] = [' '.join(le.inverse_transform(indices)) for indices in top_3_indices]

# Save submission
submission.to_csv('../data/pycaret_submission.csv', index=False)
print("PyCaret submission saved to data/pycaret_submission.csv")

# Print model performance metrics
print("\nModel Performance Metrics:")
print(pull()) 
