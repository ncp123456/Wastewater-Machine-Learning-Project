from sklearn.linear_model import LinearRegression
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

def build_linear_model():
    """Build linear regression model"""
    model = LinearRegression()
    return model

def train_and_evaluate_linear(data, target_cols):
    """Train and evaluate linear regression model"""
    print(f"\nTraining Linear Regression model for {target_cols}...")
    
    # Debug print
    print(f"Initial data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_tabular_data(
        data, target_cols
    )
    
    # Debug prints
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Build and train model
    model = build_linear_model()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('Linear Regression', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Create model_outputs directory if it doesn't exist
    os.makedirs('model_outputs', exist_ok=True)
    
    # Get processed data - using 'raw' instead of 'forecast' since we don't need time series shift
    combined_df, _, _, _, _ = process_data(data_type='raw')
    
    # Debug print
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Combined DataFrame columns: {combined_df.columns.tolist()}")
    print(f"Number of NaN values: {combined_df.isna().sum().sum()}")
    
    # Parameters - only predict Error, use facility_EFF_OP as a feature
    target_columns = ['Error']
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_linear(
        combined_df, target_columns
    )
    
    print("\nLinear Regression model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 