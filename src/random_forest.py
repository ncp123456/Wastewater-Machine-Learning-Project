from sklearn.ensemble import RandomForestRegressor
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

def build_random_forest_model():
    """Build random forest model with default parameters"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    return model

def train_and_evaluate_rf(data, target_cols):
    """Train and evaluate random forest model"""
    print(f"\nTraining Random Forest model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback=10
    )
    
    # Reshape data for random forest
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Build and train model
    model = build_random_forest_model()
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
    ResultManager.save_results('Random Forest', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_rf(
        combined_df, target_columns
    )
    
    print("\nRandom Forest model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 