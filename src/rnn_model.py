import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from file_processing import process_data
from utils import DataPreprocessor, ModelEvaluator, ResultManager
import os

def build_rnn_model(lookback, n_features):
    """Build RNN model architecture"""
    model = Sequential([
        SimpleRNN(128, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(0.3),
        SimpleRNN(64, return_sequences=True),
        Dropout(0.3),
        SimpleRNN(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_features)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_rnn(data, target_cols, lookback=10):
    """Train and evaluate RNN model"""
    print(f"\nTraining RNN model for {target_cols}...")
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_time_series_data(
        data, target_cols, lookback
    )
    
    # Build and train model
    model = build_rnn_model(lookback, len(target_cols))
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
    ResultManager.save_results('RNN', predictions, y_test, metrics)
    
    return model, history, predictions, y_test

def main():
    # Get processed data
    combined_df, _, _, _, _ = process_data(data_type='forecast')
    
    # Parameters
    target_columns = ['Error', 'facility_EFF_OP']  # Using new column names
    lookback = 10
    
    # Train and evaluate model
    model, history, predictions, actual_values = train_and_evaluate_rnn(
        combined_df, target_columns, lookback
    )
    
    print("\nRNN model training and evaluation completed.")
    print("Results have been saved and can be visualized using plotting.py")

if __name__ == "__main__":
    main() 