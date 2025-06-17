import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRanker
from sklearn.model_selection import KFold
import optuna
from sklearn.metrics import label_ranking_average_precision_score

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

# Prepare data for ranking
def prepare_ranking_data(df, is_train=True):
    # Get unique fertilizers
    fertilizers = df['Fertilizer Name'].unique()
    
    # Create ranking dataset
    ranking_data = []
    for _, row in df.iterrows():
        for fert in fertilizers:
            ranking_data.append({
                'id': row['id'],
                'Fertilizer Name': fert,
                'is_target': 1 if fert == row['Fertilizer Name'] else 0,
                **{col: row[col] for col in numeric_features + categorical_features}
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    return ranking_df

# Define objective function for Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'rank:map',  # Using pairwise ranking objective
        'tree_method': 'hist',  # Faster training
    }
    
    # Initialize cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # Prepare ranking data
    ranking_train = prepare_ranking_data(train_df)
    
    # Cross-validation loop
    for train_idx, val_idx in kf.split(ranking_train['id'].unique()):
        # Get train/val IDs
        train_ids = ranking_train['id'].unique()[train_idx]
        val_ids = ranking_train['id'].unique()[val_idx]
        
        # Split data
        train_fold = ranking_train[ranking_train['id'].isin(train_ids)]
        val_fold = ranking_train[ranking_train['id'].isin(val_ids)]
        
        # Prepare features
        X_train_fold = preprocessor.fit_transform(train_fold[numeric_features + categorical_features])
        X_val_fold = preprocessor.transform(val_fold[numeric_features + categorical_features])
        
        # Prepare ranking data
        train_groups = train_fold.groupby('id').size().values
        val_groups = val_fold.groupby('id').size().values
        
        # Create and fit model
        model = XGBRanker(**params, random_state=42)
        model.fit(
            X_train_fold,
            train_fold['is_target'],
            group=train_groups,
            eval_set=[(X_val_fold, val_fold['is_target'])],
            eval_group=[val_groups],
            verbose=False
        )
        
        # Get predictions
        val_pred = model.predict(X_val_fold)
        
        # Calculate MAP@3
        val_pred_reshaped = val_pred.reshape(-1, len(fertilizers))
        score = label_ranking_average_precision_score(
            val_fold['is_target'].values.reshape(-1, len(fertilizers)),
            val_pred_reshaped
        )
        scores.append(score)
    
    return np.mean(scores)

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Get best parameters
best_params = study.best_params
print("\nBest parameters:", best_params)

# Prepare final ranking data
ranking_train = prepare_ranking_data(train_df)
ranking_test = prepare_ranking_data(test_df, is_train=False)

# Prepare features
X_train = preprocessor.fit_transform(ranking_train[numeric_features + categorical_features])
X_test = preprocessor.transform(ranking_test[numeric_features + categorical_features])

# Train groups
train_groups = ranking_train.groupby('id').size().values
test_groups = ranking_test.groupby('id').size().values

# Create and fit final model
best_model = XGBRanker(**best_params, random_state=42)
best_model.fit(
    X_train,
    ranking_train['is_target'],
    group=train_groups,
    verbose=False
)

# Get predictions
test_pred = best_model.predict(X_test)
test_pred_reshaped = test_pred.reshape(-1, len(fertilizers))

# Get top 3 predictions for each sample
top_3_indices = np.argsort(test_pred_reshaped, axis=1)[:, -3:]
le = LabelEncoder()
le.fit(train_df['Fertilizer Name'])

# Create submission
submission = pd.DataFrame()
submission['id'] = test_df['id'].unique()
submission['Fertilizer Name'] = [' '.join(le.inverse_transform(indices)) for indices in top_3_indices]

# Save submission
submission.to_csv('../data/xgboost_baseline_submission.csv', index=False)
print("XGBoost baseline submission saved to data/xgboost_baseline_submission.csv")

# Print model performance on training data
train_pred = best_model.predict(X_train)
train_pred_reshaped = train_pred.reshape(-1, len(fertilizers))
train_map = label_ranking_average_precision_score(
    ranking_train['is_target'].values.reshape(-1, len(fertilizers)),
    train_pred_reshaped
)
print(f"\nTraining MAP@3 score: {train_map:.4f}")

# Print feature importance
feature_names = (numeric_features + 
                [f"{col}_{val}" for col, vals in 
                 zip(categorical_features, 
                     [preprocessor.named_transformers_['cat'].categories_[i] 
                      for i in range(len(categorical_features))])])
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
})
importance = importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features:")
print(importance.head(10)) 
