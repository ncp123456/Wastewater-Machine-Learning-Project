import numpy as np
import torch
from pathlib import Path
import sys
import json
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
from config import MODEL_DIR
import os
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Lag-LLaMA repository to Python path
lag_llama_path = MODEL_DIR / "lag-llama"
sys.path.append(str(lag_llama_path))

# Import Lag-LLaMA specific modules
from lag_llama.model.module import LagLlamaModel
from gluonts.torch.distributions import StudentTOutput

# Add safe globals for PyTorch 2.6
torch.serialization.add_safe_globals([StudentTOutput])

class LagLLaMAConfig:
    """Configuration class for Lag-LLaMA model"""
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = MODEL_DIR / "lag-llama" / "configs" / "lag_llama.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using default values")
            config = {}
        
        # Set configuration parameters with validation
        self.n_layer = 8  # Fixed number of layers
        self.n_head = 9  # Fixed number of attention heads
        self.n_embd_per_head = 16  # Fixed embedding per head
        self.context_length = 32  # Fixed context length
        self.dropout = 0.0  # Fixed dropout
        self.time_feat = True  # Fixed time features
        self.input_size = 92  # Fixed input size to match checkpoint
        self.max_context_length = 64  # Fixed max context length
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.patience = 10  # Early stopping patience
        self.min_delta = 1e-4  # Minimum improvement for early stopping

def validate_data(data: np.ndarray, target_cols: list) -> None:
    """Validate input data"""
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    if not all(col in data.columns for col in target_cols):
        raise ValueError(f"Target columns {target_cols} not found in data")
    
    if data.isna().any().any():
        raise ValueError("Input data contains NaN values")

def build_lag_llama_model(checkpoint_path: Path) -> Tuple[LagLlamaModel, int]:
    """Build Lag-LLaMA model and return both model and correct input size"""
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Lag-LLaMA checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint first to get the correct input size
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (PyTorch Lightning adds this)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        # Get the final feature size from the checkpoint
        final_feature_size = state_dict['transformer.wte.weight'].shape[1]
        logger.info(f"Final feature size from checkpoint: {final_feature_size}")
        
        # Calculate base input size that will result in the correct feature size
        # The formula is: final_feature_size = base_input_size * len(lags_seq) + 2 * base_input_size + 6
        # Solving for base_input_size:
        lags_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time_feat = True
        if time_feat:
            base_input_size = (final_feature_size - 6) // (len(lags_seq) + 2)
        else:
            base_input_size = final_feature_size // (len(lags_seq) + 2)
        
        logger.info(f"Calculated base input size: {base_input_size}")
        
        # Create distribution output
        distr_output = StudentTOutput()
        
        # Initialize model with the correct base input size
        model = LagLlamaModel(
            context_length=32,  # Fixed value from checkpoint
            max_context_length=64,  # Fixed value from checkpoint
            scaling="robust",
            input_size=base_input_size,  # Use the calculated base input size
            n_layer=8,  # Fixed value from checkpoint
            n_embd_per_head=16,  # Fixed value from checkpoint
            n_head=9,  # Fixed value from checkpoint
            lags_seq=lags_seq,  # Fixed lags sequence
            distr_output=distr_output,
            time_feat=time_feat,  # Fixed value from checkpoint
            dropout=0.0  # Fixed value from checkpoint
        )
        
        # Load state dict with strict=False to handle any minor mismatches
        model.load_state_dict(state_dict, strict=False)
        
        return model, base_input_size
    
    except Exception as e:
        logger.error(f"Error building Lag-LLaMA model: {str(e)}")
        raise

def train_and_evaluate_lag_llama(data: np.ndarray, target_cols: list) -> Tuple[LagLlamaModel, np.ndarray, np.ndarray]:
    """Train and evaluate Lag-LLaMA model"""
    try:
        logger.info(f"Training Lag-LLaMA model for {target_cols}...")
        
        # Validate input data
        validate_data(data, target_cols)
        
        # Initialize preprocessor and prepare data
        preprocessor = DataPreprocessor()
        
        # Prepare time series data with proper lookback
        lookback = 10  # Number of time steps to look back
        X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
            data, target_cols, lookback=lookback
        )
        
        # Print shapes for debugging
        logger.info(f"X_train shape before processing: {X_train.shape}")
        logger.info(f"X_test shape before processing: {X_test.shape}")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        # Build model and get base input size
        checkpoint_path = MODEL_DIR / "lag-llama.ckpt"
        model, base_input_size = build_lag_llama_model(checkpoint_path)
        
        # Reshape data to match model's expected input
        # First flatten the time steps and features
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Calculate expected input size based on lookback and features
        expected_input_size = lookback * len(target_cols)
        logger.info(f"Expected input size based on lookback and features: {expected_input_size}")
        
        # Ensure we match the base input size exactly
        if X_train.shape[1] != base_input_size:
            logger.warning(f"Input size {X_train.shape[1]} doesn't match model's expected base size {base_input_size}. Adjusting...")
            if X_train.shape[1] > base_input_size:
                X_train = X_train[:, :base_input_size]
                X_test = X_test[:, :base_input_size]
            else:
                pad_size = base_input_size - X_train.shape[1]
                X_train = torch.cat([X_train, torch.zeros((X_train.shape[0], pad_size))], dim=1)
                X_test = torch.cat([X_test, torch.zeros((X_test.shape[0], pad_size))], dim=1)
        
        logger.info(f"X_train shape after processing: {X_train.shape}")
        logger.info(f"X_test shape after processing: {X_test.shape}")
        
        # Set model to training mode
        model.train()
        
        # Training configuration
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = torch.nn.MSELoss()
        
        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        batch_size = config.batch_size
        n_batches = len(X_train) // batch_size
        
        for epoch in range(config.num_epochs):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss
            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss - config.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
            
            if patience_counter >= config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = []
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
    
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_lag_llama: {str(e)}")
        raise

def main():
    try:
        # Get processed data
        combined_df, _, _, _, _ = process_data(data_type='forecast')
        
        # Parameters
        target_columns = ['Error', 'facility_EFF_OP']
        
        # Train and evaluate model
        model, predictions, actual_values = train_and_evaluate_lag_llama(
            combined_df, target_columns
        )
        
        logger.info("\nLag-LLaMA model training and evaluation completed.")
        logger.info("Results have been saved and can be visualized using plotting.py")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 