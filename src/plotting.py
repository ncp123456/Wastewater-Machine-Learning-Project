import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import ResultManager
from file_processing import combined_df, date_time

class Plotter:
    def __init__(self):
        self.colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366']
        self.target_columns = ['Error', 'Observed Data']
    
    def plot_all_model_predictions(self):
        """Plot predictions from all models against actual values"""
        results = ResultManager.load_all_results()
        n_features = len(self.target_columns)
        
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 5*n_features))
        fig.suptitle('Time Series Forecasting Results - All Models')
        
        # Get actual values from any model result (they should all have the same actual values)
        actual_values = next(iter(results.values()))['actual_values']
        
        for i, col in enumerate(self.target_columns):
            ax = axes[i] if n_features > 1 else axes
            
            # Plot actual values
            ax.plot(actual_values[:, i], label='Actual', color='black', linewidth=2, alpha=0.7)
            
            # Plot predictions for each model
            for j, (model_name, result) in enumerate(results.items()):
                ax.plot(result['predictions'][:, i], 
                       label=f'{model_name} Predictions',
                       color=self.colors[j % len(self.colors)],
                       alpha=0.7)
            
            ax.set_title(f'Forecasting for {col}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('plots/all_model_predictions.png')
        plt.close()
    
    def plot_model_metrics_comparison(self):
        """Plot comparison of model performance metrics"""
        results = ResultManager.load_all_results()
        metrics = ['MSE', 'RMSE', 'R2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison')
        
        for i, metric in enumerate(metrics):
            model_values = []
            model_names = []
            
            for model_name, result in results.items():
                model_names.append(model_name)
                metric_values = [result['metrics'][f'feature_{j}'][metric] 
                               for j in range(len(self.target_columns))]
                model_values.append(metric_values)
            
            x = np.arange(len(self.target_columns))
            width = 0.8 / len(model_names)
            
            for j, values in enumerate(model_values):
                offset = width * j - width * len(model_names) / 2
                axes[i].bar(x + offset, values, width, 
                          label=model_names[j],
                          color=self.colors[j % len(self.colors)])
            
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(self.target_columns)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('plots/model_metrics_comparison.png')
        plt.close()
    
    def plot_training_history(self, history, model_name):
        """Plot training history for neural network models"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE if available
        if 'mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title(f'{model_name} - Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower()}_training_history.png')
        plt.close()
    
    def plot_original_data(self):
        """Plot original data from the dataset"""
        plt.figure(figsize=(15, 10))
        plot_cols = ['Observed Data', 'Modeled Data', 'Error']
        plot_features = combined_df[plot_cols]
        plot_features.index = date_time[:len(plot_features)]
        
        for i, col in enumerate(plot_cols):
            plt.subplot(3, 1, i+1)
            plt.plot(plot_features[col], label=col)
            plt.title(f'Original {col} Data')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/original_data.png')
        plt.close()

def plot_predictions_with_uncertainty(predictions, actual_values, uncertainty, target_cols):
    """Plot predictions with uncertainty intervals"""
    n_features = len(target_cols)
    lower_bound, upper_bound = uncertainty
    
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 5*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        # Plot actual values
        ax.plot(actual_values[:, i], 'k-', label='Actual', alpha=0.7)
        
        # Plot predictions
        ax.plot(predictions[:, i], 'r-', label='Prediction', alpha=0.7)
        
        # Plot uncertainty intervals
        ax.fill_between(
            range(len(predictions)),
            lower_bound[:, i],
            upper_bound[:, i],
            color='r',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        ax.set_title(f'Forecasting for {col}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/lag_llama_predictions.png')
    plt.close()

def main():
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize plotter
    plotter = Plotter()
    
    # Generate all plots
    print("Generating plots...")
    plotter.plot_original_data()
    plotter.plot_all_model_predictions()
    plotter.plot_model_metrics_comparison()
    
    print("Plots have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main()

