import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def scale_data(self, data):
        """Scale the data using MinMaxScaler"""
        return self.scaler.fit_transform(data)
    
    def inverse_scale(self, data):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data)
    
    def create_sequences(self, data, seq_length):
        """Create sequences for time series data"""
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:(i + seq_length)])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)
    
    def prepare_time_series_data(self, data, target_cols, lookback, test_size=0.2):
        """Prepare data for time series models"""
        scaled_data = self.scale_data(data[target_cols])
        X, y = self.create_sequences(scaled_data, lookback)
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_tabular_data(self, data, target_cols, feature_cols=None, test_size=0.2):
        """Prepare data for traditional ML models"""
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col not in target_cols]
            
        X = data[feature_cols]
        y = data[target_cols]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate common evaluation metrics"""
        metrics = {}
        
        # Convert to NumPy arrays if they're pandas objects
        y_true = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred = y_pred.values if hasattr(y_pred, 'values') else y_pred
        
        # Handle both single and multi-dimensional targets
        if len(y_true.shape) > 1:
            n_features = y_true.shape[1]
            for i in range(n_features):
                y_t = y_true[:, i]
                y_p = y_pred[:, i]
                
                mse = mean_squared_error(y_t, y_p)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_t, y_p)
                
                metrics[f'feature_{i}'] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                }
        else:
            # Single target variable
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            metrics['target'] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
        return metrics

class ResultManager:
    @staticmethod
    def save_results(model_name, predictions, actual_values, metrics):
        """Save model results in a standardized format"""
        results = {
            'model_name': model_name,
            'predictions': predictions,
            'actual_values': actual_values,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        }
        
        # Save to a central results directory
        joblib.dump(results, f'model_outputs/{model_name}_results.joblib')
        
    @staticmethod
    def load_all_results():
        """Load all model results"""
        import glob
        results = {}
        for file in glob.glob('model_outputs/*_results.joblib'):
            model_results = joblib.load(file)
            results[model_results['model_name']] = model_results
        return results 