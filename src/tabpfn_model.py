import numpy as np
from tabpfn import TabPFNClassifier
import torch
from sklearn.preprocessing import KBinsDiscretizer
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
from config import DEVICE, BATCH_SIZE, NUM_EPOCHS
import os

class TabPFNRegressor:
    """Wrapper for TabPFN to handle regression through classification"""
    def __init__(self, n_bins=100, device='cpu'):
        self.n_bins = n_bins
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        self.classifier = TabPFNClassifier(device=device)
        
    def fit(self, X, y):
        """Fit the model by discretizing y into classes"""
        y_discrete = self.discretizer.fit_transform(y.reshape(-1, 1))
        self.classifier.fit(X, y_discrete.ravel())
        return self
        
    def predict(self, X):
        """Predict and convert back to continuous values"""
        class_probs = self.classifier.predict_proba(X)
        # Get expected value using class probabilities
        expected_class = np.sum(class_probs * np.arange(self.n_bins), axis=1)
        # Convert back to original scale
        predictions = self.discretizer.inverse_transform(expected_class.reshape(-1, 1))
        return predictions

def train_and_evaluate_tabpfn(data, target_cols):
    """Train and evaluate TabPFN model"""
    print(f"\nTraining TabPFN model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback=10
    )
    
    # Reshape data for TabPFN
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Train separate models for each target
    all_predictions = []
    all_y_test = []
    
    for i, target_col in enumerate(target_cols):
        print(f"\nTraining for target: {target_col}")
        
        # Get single target
        y_train_single = y_train[:, i]
        y_test_single = y_test[:, i]
        
        # Convert to classification problem (bin the continuous values)
        n_bins = 10
        y_train_binned = np.digitize(y_train_single, np.linspace(y_train_single.min(), y_train_single.max(), n_bins))
        y_test_binned = np.digitize(y_test_single, np.linspace(y_test_single.min(), y_test_single.max(), n_bins))
        
        # Build and train model
        model = TabPFNClassifier(device='cpu')
        model.fit(X_train, y_train_binned)
        
        # Make predictions
        predictions_binned = model.predict(X_test)
        
        # Convert back to continuous values
        bin_centers = np.linspace(y_test_single.min(), y_test_single.max(), n_bins)
        predictions = bin_centers[predictions_binned]
        
        all_predictions.append(predictions)
        all_y_test.append(y_test_single)
    
    # Stack predictions and actual values
    predictions = np.column_stack(all_predictions)
    y_test = np.column_stack(all_y_test)
    
    # Inverse transform predictions and actual values
    predictions = preprocessor.inverse_scale(predictions)
    y_test = preprocessor.inverse_scale(y_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('TabPFN', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_tabpfn(
        combined_df, target_columns
    )
    
    print("\nTabPFN model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 