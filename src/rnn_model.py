import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from file_processing import process_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def clean_data(data, target_cols):
    """Clean and prepare data for training"""
    # Ensure all columns are numeric
    data = data.select_dtypes(include=[np.number])
    
    # Calculate NaN percentages for each column
    nan_percentages = data.isna().mean() * 100
    print("\nNaN percentages by column:")
    for col, percentage in nan_percentages.items():
        print(f"{col}: {percentage:.2f}%")
    
    # Drop columns with 100% NaN values
    columns_to_drop = nan_percentages[nan_percentages == 100].index
    if len(columns_to_drop) > 0:
        print(f"\nDropping columns with 100% NaN values: {columns_to_drop.tolist()}")
        data = data.drop(columns=columns_to_drop)
    
    # Fill remaining NaN values with column medians
    data = data.fillna(data.median())
    
    # Verify no NaN values remain
    if data.isna().any().any():
        raise ValueError("Data still contains NaN values after cleaning")
    
    return data

def train_and_evaluate_rnn(data, target_cols):
    """Train and evaluate RNN model"""
    # Clean and prepare data
    print("\nInitial data check:")
    print(f"Data shape: {data.shape}")
    print(f"Number of NaN values: {data.isna().sum().sum()}")
    print(f"Columns with NaN values: {data.columns[data.isna().any()].tolist()}")
    
    data = clean_data(data, target_cols)
    
    print("\nAfter cleaning:")
    print(f"Data shape: {data.shape}")
    print(f"Number of NaN values: {data.isna().sum().sum()}")
    
    # Check for infinite values
    if np.isinf(data.values).any():
        raise ValueError("Input data contains infinite values")
    
    print(f"\nData shape before splitting: {data.shape}")
    
    # Split data
    X = data.drop(columns=target_cols)
    y = data[target_cols].values.ravel()  # Convert to 1D array
    
    print("\nAfter splitting:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X NaN count: {X.isna().sum().sum()}")
    print(f"y NaN count: {np.isnan(y).sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Reshape data for RNN
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print("\nAfter reshaping:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    print("\nAfter tensor conversion:")
    print(f"X_train NaN count: {torch.isnan(X_train).sum().item()}")
    print(f"X_test NaN count: {torch.isnan(X_test).sum().item()}")
    print(f"y_train NaN count: {torch.isnan(y_train).sum().item()}")
    print(f"y_test NaN count: {torch.isnan(y_test).sum().item()}")
    
    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = RNNModel(input_size)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"Warning: NaN detected in model outputs at epoch {epoch}")
                print(f"Outputs: {outputs}")
                print(f"Input batch shape: {batch_X.shape}")
                print(f"Input batch NaN count: {torch.isnan(batch_X).sum().item()}")
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    print("\nAfter model evaluation:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions NaN count: {torch.isnan(predictions).sum().item()}")
    
    # Convert predictions and actual values to numpy arrays
    predictions = predictions.numpy()
    actual_values = y_test.numpy()
    
    print("\nAfter numpy conversion:")
    print(f"Predictions NaN count: {np.isnan(predictions).sum()}")
    print(f"Actual values NaN count: {np.isnan(actual_values).sum()}")
    
    # Check for NaN in predictions
    if np.isnan(predictions).any():
        print("Warning: Predictions contain NaN values. Replacing with median...")
        predictions = np.nan_to_num(predictions, nan=np.nanmedian(predictions))
    
    return predictions, actual_values, model

def process_combined_df():
    """Process and return results for combined_df"""
    # Get data
    combined_df, date_time, _, _, _, _, _ = process_data(data_type='forecast')
    
    # Define target columns
    target_cols = ['Error']  # Adjust based on your needs
    
    # Train and evaluate model
    predictions, actual_values, model = train_and_evaluate_rnn(combined_df, target_cols)
    
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
    predictions, actual_values, model = train_and_evaluate_rnn(timeseries_dataframe, target_cols)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_values)
    
    # Plot results
    plot_results(predictions, actual_values, timeseries_dataframe.index[-len(predictions):], 'timeseries')
    
    return metrics, predictions, actual_values, timeseries_dataframe

def calculate_metrics(predictions, actual_values):
    """Calculate and return evaluation metrics"""
    # Ensure arrays are 1D
    predictions = predictions.reshape(-1)
    actual_values = actual_values.reshape(-1)
    
    # Check for NaN values
    if np.isnan(predictions).any() or np.isnan(actual_values).any():
        raise ValueError("Cannot calculate metrics with NaN values")
    
    metrics = {
        'mse': mean_squared_error(actual_values, predictions),
        'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
        'mae': mean_absolute_error(actual_values, predictions),
        'r2': r2_score(actual_values, predictions)
    }
    return metrics

def plot_results(predictions, actual_values, dates, data_type):
    """Plot and save results"""
    # Ensure arrays are 1D
    predictions = predictions.reshape(-1)
    actual_values = actual_values.reshape(-1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_values, label='Actual', color='blue')
    plt.plot(dates, predictions, label='Predicted', color='red')
    plt.title(f'RNN Predictions vs Actual Values ({data_type})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/rnn_{data_type}_predictions.png')
    plt.close()

def save_results(metrics, predictions, actual_values, dates, data_type):
    """Save results to output folder"""
    os.makedirs('output', exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'output/rnn_{data_type}_metrics.csv', index=False)
    
    # Ensure arrays are 1-dimensional
    predictions = predictions.reshape(-1)
    actual_values = actual_values.reshape(-1)
    
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
    results_df.to_csv(f'output/rnn_{data_type}_results.csv', index=False)

def main():
    """Main execution function"""
    print("\nProcessing combined_df with RNN...")
    combined_metrics, combined_predictions, combined_actual, date_time = process_combined_df()
    save_results(combined_metrics, combined_predictions, combined_actual, 
                date_time[-len(combined_predictions):], 'combined_df')
    
    print("\nProcessing timeseries_dataframe with RNN...")
    timeseries_metrics, timeseries_predictions, timeseries_actual, timeseries_dataframe = process_timeseries_dataframe()
    save_results(timeseries_metrics, timeseries_predictions, timeseries_actual,
                timeseries_dataframe.index[-len(timeseries_predictions):], 'timeseries')
    
    # Print summary of results
    print("\nRNN Results Summary:")
    print("\nCombined DataFrame Results:")
    for metric, value in combined_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTimeseries DataFrame Results:")
    for metric, value in timeseries_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main() 