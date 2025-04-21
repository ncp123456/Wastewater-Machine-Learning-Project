import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

def build_lstm_model(lookback, n_features):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(128, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_features)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_lstm(data, target_cols, lookback=10):
    """Train and evaluate LSTM model"""
    print(f"\nTraining LSTM model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback
    )
    
    # Build and train model
    model = build_lstm_model(lookback, len(target_cols))
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = preprocessor.inverse_scale(predictions)
    y_test = preprocessor.inverse_scale(y_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test, predictions)
    
    # Save results
    ResultManager.save_results('LSTM', predictions, y_test, metrics)
    
    return model, history, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    lookback = 10
    
    # Train and evaluate model
    model, history, predictions, actual_values = train_and_evaluate_lstm(
        combined_df, target_columns, lookback
    )
    
    print("\nLSTM model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 