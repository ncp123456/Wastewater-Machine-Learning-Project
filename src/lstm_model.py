import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from file_processing import process_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_sequences(data, target_cols, sequence_length=10):
    """Prepare sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(data.iloc[i + sequence_length][target_cols].values)
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(data, target_cols):
    """Train and evaluate LSTM model"""
    # Prepare sequences
    X, y = prepare_sequences(data, target_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = len(target_cols)
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Convert to numpy and ensure 1D arrays
    predictions = predictions.numpy().flatten()
    y_test = y_test.numpy().flatten()
    
    return predictions, y_test

def process_combined_df():
    """Process and return results for combined_df"""
    # Get data
    combined_df, date_time, _, _, _, _, _ = process_data(data_type='forecast')
    
    # Define target columns
    target_cols = ['Error']  # Adjust based on your needs
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_lstm(combined_df, target_cols)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_values)
    
    # Plot results
    plot_results(predictions, actual_values, date_time[-len(predictions):], 'combined_df')
    
    return metrics, predictions, actual_values, date_time

def process_timeseries_dataframe():
    """Process and return results for timeseries_dataframe"""
    # Get data
    _, _, _, _, _, timeseries_dataframe, _ = process_data()
    
    # Clean the data
    print("\nCleaning timeseries data...")
    print(f"Initial shape: {timeseries_dataframe.shape}")
    print(f"Initial NaN count: {timeseries_dataframe.isna().sum().sum()}")
    
    # Calculate NaN percentages for each column
    nan_percentages = (timeseries_dataframe.isna().sum() / len(timeseries_dataframe)) * 100
    print("\nNaN percentages by column:")
    for col, percentage in nan_percentages.items():
        if percentage > 0:
            print(f"{col}: {percentage:.2f}%")
    
    # Drop columns that are 100% NaN
    columns_to_drop = nan_percentages[nan_percentages == 100].index
    if len(columns_to_drop) > 0:
        print(f"\nDropping columns with 100% NaN values: {columns_to_drop.tolist()}")
        timeseries_dataframe = timeseries_dataframe.drop(columns=columns_to_drop)
    
    # Fill remaining NaN values with column medians
    timeseries_dataframe = timeseries_dataframe.fillna(timeseries_dataframe.median())
    
    # Verify no NaN values remain
    remaining_nans = timeseries_dataframe.isna().sum().sum()
    if remaining_nans > 0:
        print(f"\nWarning: {remaining_nans} NaN values remain after cleaning")
        print("Columns with remaining NaN values:")
        for col in timeseries_dataframe.columns[timeseries_dataframe.isna().any()]:
            print(f"{col}: {timeseries_dataframe[col].isna().sum()} NaN values")
    else:
        print("\nAll NaN values have been cleaned")
    
    print(f"Final shape: {timeseries_dataframe.shape}")
    
    # Define target columns
    target_cols = ['EFF_OP']  # Adjust based on your needs
    
    # Train and evaluate model
    predictions, actual_values = train_and_evaluate_lstm(timeseries_dataframe, target_cols)
    
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
    plt.title(f'LSTM Predictions vs Actual Values ({data_type})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/lstm_{data_type}_predictions.png')
    plt.close()

def save_results(metrics, predictions, actual_values, dates, data_type):
    """Save results to output folder"""
    os.makedirs('output', exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'output/lstm_{data_type}_metrics.csv', index=False)
    
    # Save predictions and actual values
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actual_values,
        'predicted': predictions
    })
    results_df.to_csv(f'output/lstm_{data_type}_results.csv', index=False)

def main():
    """Main execution function"""
    print("\nProcessing combined_df with LSTM...")
    combined_metrics, combined_predictions, combined_actual, date_time = process_combined_df()
    save_results(combined_metrics, combined_predictions, combined_actual, 
                date_time[-len(combined_predictions):], 'combined_df')
    
    print("\nProcessing timeseries_dataframe with LSTM...")
    timeseries_metrics, timeseries_predictions, timeseries_actual, timeseries_dataframe = process_timeseries_dataframe()
    save_results(timeseries_metrics, timeseries_predictions, timeseries_actual,
                timeseries_dataframe.index[-len(timeseries_predictions):], 'timeseries')
    
    # Print summary of results
    print("\nLSTM Results Summary:")
    print("\nCombined DataFrame Results:")
    for metric, value in combined_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTimeseries DataFrame Results:")
    for metric, value in timeseries_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main() 