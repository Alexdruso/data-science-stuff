import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Read the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Prepare features
numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit the model
X_train = train_df[numeric_features + categorical_features]
y_train = train_df['Fertilizer Name']
pipeline.fit(X_train, y_train)

# Get probability predictions
X_test = test_df[numeric_features + categorical_features]
y_pred_proba = pipeline.predict_proba(X_test)

# Get top 3 predictions for each sample
top_3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:]
le = LabelEncoder()
le.fit(train_df['Fertilizer Name'])

# Create submission
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['Fertilizer Name'] = [' '.join(le.inverse_transform(indices)) for indices in top_3_indices]

# Save submission
submission.to_csv('../data/logistic_baseline_submission.csv', index=False)
print("Logistic regression baseline submission saved to data/logistic_baseline_submission.csv")

# Print model performance on training data
train_pred_proba = pipeline.predict_proba(X_train)
train_top_3_indices = np.argsort(train_pred_proba, axis=1)[:, -3:]
train_predictions = [' '.join(le.inverse_transform(indices)) for indices in train_top_3_indices]
train_accuracy = sum(1 for true, pred in zip(y_train, train_predictions) if true in pred.split()) / len(y_train)
print(f"\nTraining accuracy (true label in top 3): {train_accuracy:.4f}") 
