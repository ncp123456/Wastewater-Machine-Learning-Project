import numpy as np
import torch
from pathlib import Path
import sys
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
from config import MODEL_DIR
import os

# Add Lag-LLaMA repository to Python path
lag_llama_path = MODEL_DIR / "lag-llama"
sys.path.append(str(lag_llama_path))

# Import Lag-LLaMA specific modules
from lag_llama.model import LagLLaMA
from lag_llama.config import LagLLaMAConfig

def build_lag_llama_model():
    """Build Lag-LLaMA model"""
    # Load the pre-trained model
    checkpoint_path = MODEL_DIR / "lag-llama.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lag-LLaMA checkpoint not found at {checkpoint_path}")
    
    # Initialize model configuration
    config = LagLLaMAConfig(
        n_layer=8,
        n_head=8,
        n_embd=512,
        block_size=1024,
        bias=False,
        vocab_size=None,  # Will be set by the model
        dropout=0.1
    )
    
    # Create model instance
    model = LagLLaMA(config)
    
    # Load pre-trained weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model

def train_and_evaluate_lag_llama(data, target_cols):
    """Train and evaluate Lag-LLaMA model"""
    print(f"\nTraining Lag-LLaMA model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback=10
    )
    
    # Convert to PyTorch tensors and ensure correct shape
    X_train = torch.FloatTensor(X_train).transpose(1, 2)  # [batch, features, seq_len]
    X_test = torch.FloatTensor(X_test).transpose(1, 2)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Build model
    model = build_lag_llama_model()
    model.train()
    
    # Training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    batch_size = 32
    n_batches = len(X_train) // batch_size
    for epoch in range(100):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/n_batches:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = []
        batch_size = 32
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i+batch_size]
            batch_pred = model(batch_x)
            predictions.append(batch_pred)
        predictions = torch.cat(predictions, dim=0).numpy()
    
    # Convert back to numpy
    y_test = y_test.numpy()
    
    # Inverse transform predictions and actual values
    predictions = preprocessor.inverse_scale(predictions)
    y_test = preprocessor.inverse_scale(y_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('Lag-LLaMA', predictions, y_test, metrics)
    
    return model, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    
    # Train and evaluate model
    model, predictions, actual_values = train_and_evaluate_lag_llama(
        combined_df, target_columns
    )
    
    print("\nLag-LLaMA model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 