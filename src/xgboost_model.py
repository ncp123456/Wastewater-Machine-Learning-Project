import xgboost as xgb
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

def build_xgboost_model():
    """Build XGBoost model with default parameters"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    return model

def train_and_evaluate_xgboost(data, target_cols):
    """Train and evaluate XGBoost model"""
    print(f"\nTraining XGBoost model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback=10
    )
    
    # Reshape data for XGBoost
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Build and train model
    model = build_xgboost_model()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = preprocessor.inverse_scale(predictions)
    y_test = preprocessor.inverse_scale(y_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('XGBoost', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_xgboost(
        combined_df, target_columns
    )
    
    print("\nXGBoost model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 