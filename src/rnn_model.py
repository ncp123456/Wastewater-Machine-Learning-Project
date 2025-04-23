import numpy as np
import torch
import torch.nn as nn
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, input_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Apply fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_and_evaluate_rnn(data, target_cols, lookback=10):
    """Train and evaluate RNN model"""
    print(f"\nTraining RNN model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Initialize model
    model = RNNModel(input_size=len(target_cols))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    batch_size = 32
    n_epochs = 100
    n_batches = len(X_train) // batch_size
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_batches:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Convert back to numpy arrays
    predictions = predictions.numpy()
    y_test = y_test.numpy()
    
    # Inverse transform predictions and actual values
    predictions = preprocessor.inverse_scale(predictions)
    y_test = preprocessor.inverse_scale(y_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('RNN', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    lookback = 10
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_rnn(
        combined_df, target_columns, lookback
    )
    
    print("\nRNN model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 