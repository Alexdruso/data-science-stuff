import polars as pl
from pycaret.regression import *
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load and preprocess the data using Polars."""
    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")
    
    # Basic preprocessing
    # Convert date columns to datetime if they exist
    date_columns = [col for col in train_df.columns if 'date' in col.lower()]
    for col in date_columns:
        train_df = train_df.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d"))
        test_df = test_df.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d"))
    
    return train_df, test_df

def setup_pycaret_experiment(train_df):
    """Set up PyCaret experiment."""
    # Convert Polars DataFrame to Pandas for PyCaret
    train_pd = train_df.to_pandas()
    
    # Initialize PyCaret experiment
    exp = setup(
        data=train_pd,
        target='Calories',  # Replace with actual target column name
        session_id=RANDOM_SEED,
        normalize=True,
        feature_selection=True,
    )
    
    return exp

def train_and_predict(exp, test_df):
    """Train models and generate predictions."""
    # Compare models
    best_model = compare_models(
        n_select=1,
        sort='RMSE'  # Using RMSE as metric
    )
    
    # Finalize the model
    final_model = finalize_model(best_model)
    
    # Generate predictions
    test_pd = test_df.to_pandas()
    predictions = predict_model(final_model, data=test_pd)
    
    return predictions

def create_submission(predictions, test_df):
    """Create submission file."""
    submission = pl.DataFrame({
        'id': test_df['id'],
        'target': predictions['prediction_label']
    })
    
    # Create timestamp for submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submissions/submission_{timestamp}.csv"
    
    # Save submission
    submission.write_csv(submission_path)
    print(f"Submission saved to {submission_path}")

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Setting up PyCaret experiment...")
    exp = setup_pycaret_experiment(train_df)
    
    print("Training models and generating predictions...")
    predictions = train_and_predict(exp, test_df)
    
    print("Creating submission file...")
    create_submission(predictions, test_df)
    
    print("Done!")

if __name__ == "__main__":
    main() 
