import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier
from file_processing import process_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_tabpfn(data, target_cols):
    """Train and evaluate TabPFN model"""
    # Split data
    X = data.drop(columns=target_cols)
    y = data[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to classification problem (bin the continuous values)
    n_bins = 10
    # Ensure y_train and y_test are 1D arrays
    y_train_1d = y_train.values.ravel()
    y_test_1d = y_test.values.ravel()
    
    # Create bins based on the range of the data
    bins = np.linspace(y_train_1d.min(), y_train_1d.max(), n_bins + 1)
    
    # Bin the data
    y_train_binned = np.digitize(y_train_1d, bins) - 1  # Subtract 1 to get 0-based indices
    y_test_binned = np.digitize(y_test_1d, bins) - 1
    
    # Train model
    model = TabPFNClassifier(device='cpu')
    model.fit(X_train, y_train_binned)
    
    # Make predictions
    predictions_binned = model.predict(X_test)
    
    # Convert back to continuous values using bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    predictions = bin_centers[predictions_binned]
    
    return predictions, y_test_1d

def process_combined_df():
    """Process and return results for combined_df"""
    # Get data
    combined_df, date_time, _, _, _, _, _ = process_data(data_type='forecast')
    
    # Define target columns
    target_cols = ['Error']  # Adjust based on your needs
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_tabpfn(combined_df, target_cols)
    
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
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_tabpfn(timeseries_dataframe, target_cols)
    
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
    plt.title(f'TabPFN Predictions vs Actual Values ({data_type})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/tabpfn_{data_type}_predictions.png')
    plt.close()

def save_results(metrics, predictions, actual_values, dates, data_type):
    """Save results to output folder"""
    os.makedirs('output', exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'output/tabpfn_{data_type}_metrics.csv', index=False)
    
    # Save predictions and actual values
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual_values,
        'predicted': predictions
    })
    results_df.to_csv(f'output/tabpfn_{data_type}_results.csv', index=False)

def main():
    """Main execution function"""
    print("\nProcessing combined_df with TabPFN...")
    combined_metrics, combined_predictions, combined_actual, date_time = process_combined_df()
    save_results(combined_metrics, combined_predictions, combined_actual, 
                date_time[-len(combined_predictions):], 'combined_df')
    
    print("\nProcessing timeseries_dataframe with TabPFN...")
    timeseries_metrics, timeseries_predictions, timeseries_actual, timeseries_dataframe = process_timeseries_dataframe()
    save_results(timeseries_metrics, timeseries_predictions, timeseries_actual,
                timeseries_dataframe.index[-len(timeseries_predictions):], 'timeseries')
    
    # Print summary of results
    print("\nTabPFN Results Summary:")
    print("\nCombined DataFrame Results:")
    for metric, value in combined_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTimeseries DataFrame Results:")
    for metric, value in timeseries_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main() 