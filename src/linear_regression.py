import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from file_processing import process_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_linear_regression(data, target_cols):
    """Train and evaluate Linear Regression model"""
    # Print initial data information
    print("\nDebug information for model training:")
    print(f"Initial data shape: {data.shape}")
    print(f"Target columns: {target_cols}")
    
    # Ensure all columns are numeric
    data = data.select_dtypes(include=[np.number])
    print(f"After selecting numeric columns: {data.shape}")
    
    # Check if target columns exist in the data
    missing_targets = [col for col in target_cols if col not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")
    
    # Drop rows with NaN values in target columns
    data = data.dropna(subset=target_cols)
    print(f"After dropping NaN in target columns: {data.shape}")
    
    # Check for infinite values
    if np.isinf(data.values).any():
        raise ValueError("Input data contains infinite values")
    
    print(f"\nData shape before splitting: {data.shape}")
    
    # Split data with a smaller test size to use more data for evaluation
    X = data.drop(columns=target_cols)
    y = data[target_cols]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return predictions, y_test.values

def process_combined_df():
    """Process and return results for combined_df"""
    # Get data
    combined_df, date_time, _, _, _, _, _ = process_data(data_type='forecast')
    
    # Define target columns
    target_cols = ['Error']  # Adjust based on your needs
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_linear_regression(combined_df, target_cols)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_values)
    
    # Plot results
    plot_results(predictions, actual_values, date_time[-len(predictions):], 'combined_df')
    
    return metrics, predictions, actual_values, date_time

def process_timeseries_dataframe():
    """Process and return results for timeseries_dataframe"""
    # Get data
    _, _, _, _, _, timeseries_dataframe, _ = process_data()
    
    # Define target columns
    target_cols = ['EFF_OP']  # Adjust based on your needs
    
    # Ensure we only use numeric columns
    timeseries_dataframe = timeseries_dataframe.select_dtypes(include=[np.number])
    
    # Print initial column information
    print("\nInitial column information:")
    print(f"Total columns: {len(timeseries_dataframe.columns)}")
    print(f"Columns with NaN values: {timeseries_dataframe.columns[timeseries_dataframe.isna().any()].tolist()}")
    
    # Calculate percentage of NaN values in each column
    nan_percentages = (timeseries_dataframe.isna().sum() / len(timeseries_dataframe)) * 100
    print("\nNaN percentages by column:")
    for col, percent in nan_percentages[nan_percentages > 0].items():
        print(f"{col}: {percent:.2f}%")
    
    # Drop columns that are more than 50% NaN
    columns_to_drop = nan_percentages[nan_percentages > 50].index
    timeseries_dataframe = timeseries_dataframe.drop(columns=columns_to_drop)
    print(f"\nDropped columns with >50% NaN values: {list(columns_to_drop)}")
    
    # Fill remaining NaN values with column medians (more robust than mean)
    timeseries_dataframe = timeseries_dataframe.fillna(timeseries_dataframe.median())
    
    # Verify no NaN values remain
    remaining_nans = timeseries_dataframe.isna().sum().sum()
    print(f"\nNaN values after processing: {remaining_nans}")
    if remaining_nans > 0:
        print("Columns still containing NaN values:")
        for col in timeseries_dataframe.columns[timeseries_dataframe.isna().any()]:
            print(f"{col}: {timeseries_dataframe[col].isna().sum()} NaN values")
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_linear_regression(timeseries_dataframe, target_cols)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_values)
    
    # Plot results
    plot_results(predictions, actual_values, timeseries_dataframe.index[-len(predictions):], 'timeseries')
    
    return metrics, predictions, actual_values, timeseries_dataframe

def calculate_metrics(predictions, actual_values):
    """Calculate and return evaluation metrics"""
    metrics = {
        'mse': mean_squared_error(actual_values, predictions),
        'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
        'mae': mean_absolute_error(actual_values, predictions),
        'r2': r2_score(actual_values, predictions)
    }
    return metrics

def plot_results(predictions, actual_values, dates, data_type):
    """Plot and save results"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_values, label='Actual')
    plt.plot(dates, predictions, label='Predicted')
    plt.title(f'Linear Regression Predictions vs Actual Values ({data_type})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/linear_regression_{data_type}_predictions.png')
    plt.close()

def save_results(metrics, predictions, actual_values, dates, data_type):
    """Save results to output folder"""
    os.makedirs('output', exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'output/linear_regression_{data_type}_metrics.csv', index=False)
    
    # Ensure arrays are 1-dimensional
    predictions = predictions.reshape(-1) if len(predictions.shape) > 1 else predictions
    actual_values = actual_values.reshape(-1) if len(actual_values.shape) > 1 else actual_values
    
    # Ensure dates and predictions have same length
    if len(dates) != len(predictions):
        min_len = min(len(dates), len(predictions))
        dates = dates[:min_len]
        predictions = predictions[:min_len]
        actual_values = actual_values[:min_len]
    
    # Save predictions and actual values
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual_values,
        'predicted': predictions
    })
    results_df.to_csv(f'output/linear_regression_{data_type}_results.csv', index=False)

def main():
    """Main execution function"""
    print("\nProcessing combined_df with Linear Regression...")
    combined_metrics, combined_predictions, combined_actual, date_time = process_combined_df()
    save_results(combined_metrics, combined_predictions, combined_actual, 
                date_time[-len(combined_predictions):], 'combined_df')
    
    print("\nProcessing timeseries_dataframe with Linear Regression...")
    timeseries_metrics, timeseries_predictions, timeseries_actual, timeseries_dataframe = process_timeseries_dataframe()
    save_results(timeseries_metrics, timeseries_predictions, timeseries_actual,
                timeseries_dataframe.index[-len(timeseries_predictions):], 'timeseries')
    
    # Print summary of results
    print("\nLinear Regression Results Summary:")
    print("\nCombined DataFrame Results:")
    for metric, value in combined_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTimeseries DataFrame Results:")
    for metric, value in timeseries_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main() 