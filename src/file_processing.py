import pandas as pd
import os
import numpy as np

# Define what variables can be imported from this file
__all__ = ['combined_df', 'date_time', 'Error_data_mean', 'Error_data_std', 'facility_df', 'sumo1b_df', 
          'normalization_stats', 'timeseries_dataframe', 'timeseries_normalization_stats']

def normalize_dataframe(df, stats_dict=None):
    """
    Normalize a dataframe using mean and standard deviation.
    If stats_dict is provided, use those statistics, otherwise calculate new ones.
    
    Args:
        df (pd.DataFrame): DataFrame to normalize
        stats_dict (dict, optional): Dictionary containing mean and std for each column
    
    Returns:
        tuple: (normalized_df, stats_dict)
    """
    if stats_dict is None:
        stats_dict = {}
        for col in df.columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    normalized_df = df.copy()
    for col in df.columns:
        mean = stats_dict[col]['mean']
        std = stats_dict[col]['std']
        normalized_df[col] = (df[col] - mean) / std
    
    return normalized_df, stats_dict

def process_data(data_type='all'):
    """
    Process and return data based on the specified type.
    
    Args:
        data_type (str): Type of data to return. Options:
            - 'all': Returns all processed data
            - 'forecast': Returns data suitable for forecasting
            - 'correction': Returns data suitable for error correction
            - 'raw': Returns raw processed data without additional transformations
    
    Returns:
        tuple: Depending on data_type, returns different combinations of:
            - combined_df: Main processed DataFrame
            - date_time: DateTime series
            - Error_data_mean: Mean of error
            - Error_data_std: Standard deviation of error
            - normalization_stats: Dictionary containing mean and std for all columns
            - timeseries_dataframe: DataFrame containing all facility data
            - timeseries_normalization_stats: Dictionary containing mean and std for timeseries data
    """
    # Read CSV files
    facility_df = pd.read_csv('src/facility (4).csv')
    sumo1b_df = pd.read_csv('src/sumo1b (2).csv')
    
    print(f"Initial facility_df shape: {facility_df.shape}")
    print(f"Initial sumo1b_df shape: {sumo1b_df.shape}")

    # Remove rows with NaN values in key columns
    facility_df = facility_df.dropna(subset=['EFF_OP'])
    sumo1b_df = sumo1b_df.dropna(subset=['EFF_OP'])
    
    print(f"After dropping NaN in EFF_OP - facility_df shape: {facility_df.shape}")
    print(f"After dropping NaN in EFF_OP - sumo1b_df shape: {sumo1b_df.shape}")

    # Set datetime index for both dataframes
    facility_df['MeasurementDate'] = pd.to_datetime(facility_df['MeasurementDate'])
    facility_df.set_index('MeasurementDate', inplace=True)
    
    sumo1b_df['Date'] = pd.to_datetime(sumo1b_df['Date'])
    sumo1b_df.set_index('Date', inplace=True)
    
    print(f"After setting index - facility_df shape: {facility_df.shape}")
    print(f"After setting index - sumo1b_df shape: {sumo1b_df.shape}")
    print(f"facility_df index: {facility_df.index[:5]}")
    print(f"sumo1b_df index: {sumo1b_df.index[:5]}")

    # Resample both dataframes to daily frequency
    facility_df = facility_df.resample('D').mean()
    sumo1b_df = sumo1b_df.resample('D').mean()
    
    print(f"After resampling - facility_df shape: {facility_df.shape}")
    print(f"After resampling - sumo1b_df shape: {sumo1b_df.shape}")

    # Create timeseries_dataframe with all facility data
    timeseries_dataframe = facility_df.copy()

    # Print debug information
    print("\nDebug information for timeseries_dataframe:")
    print(f"Initial shape: {timeseries_dataframe.shape}")
    print(f"Number of NaN values: {timeseries_dataframe.isna().sum().sum()}")
    print(f"Columns with NaN values: {timeseries_dataframe.columns[timeseries_dataframe.isna().any()].tolist()}")

    # Fill NaN values with column means, but only for columns that have numeric data
    numeric_cols = timeseries_dataframe.select_dtypes(include=[np.number]).columns
    timeseries_dataframe[numeric_cols] = timeseries_dataframe[numeric_cols].fillna(timeseries_dataframe[numeric_cols].mean())

    # Verify no NaN values remain
    print("\nVerification after filling NaN values:")
    print(f"Shape: {timeseries_dataframe.shape}")
    print(f"Number of NaN values: {timeseries_dataframe.isna().sum().sum()}")
    print(f"Columns with NaN values: {timeseries_dataframe.columns[timeseries_dataframe.isna().any()].tolist()}")
    
    if timeseries_dataframe.isna().any().any():
        raise ValueError("timeseries_dataframe still contains NaN values after filling")

    # Save original timeseries_dataframe before normalization
    original_timeseries_dataframe = timeseries_dataframe.copy()

    # Normalize timeseries_dataframe
    timeseries_dataframe, timeseries_normalization_stats = normalize_dataframe(timeseries_dataframe)

    # Rename columns to indicate source
    facility_df.columns = [f'facility_{col}' for col in facility_df.columns]
    sumo1b_df.columns = [f'sumo_{col}' for col in sumo1b_df.columns]

    # Find the overlapping date range
    start_date = max(facility_df.index.min(), sumo1b_df.index.min())
    end_date = min(facility_df.index.max(), sumo1b_df.index.max())
    
    print(f"Overlapping date range: {start_date} to {end_date}")
    
    # Filter both dataframes to the overlapping date range
    facility_df = facility_df[start_date:end_date]
    sumo1b_df = sumo1b_df[start_date:end_date]
    
    print(f"After filtering to overlapping dates - facility_df shape: {facility_df.shape}")
    print(f"After filtering to overlapping dates - sumo1b_df shape: {sumo1b_df.shape}")

    # Select only the columns we need for the model
    facility_cols = ['facility_EFF_OP', 'facility_EFF_TSS', 'facility_EFF_NHX', 'facility_EFF_TP']
    sumo_cols = ['sumo_EFF_OP', 'sumo_EFF_TSS', 'sumo_EFF_NHX', 'sumo_EFF_TP']
    
    facility_df = facility_df[facility_cols]
    sumo1b_df = sumo1b_df[sumo_cols]
    
    print(f"After selecting columns - facility_df shape: {facility_df.shape}")
    print(f"After selecting columns - sumo1b_df shape: {sumo1b_df.shape}")

    # Calculate error series before combining
    error = sumo1b_df['sumo_EFF_OP'] - facility_df['facility_EFF_OP']
    
    print(f"Error shape: {error.shape}")
    print(f"Number of NaN in error: {error.isna().sum()}")
    
    # Combine DataFrames and error in one operation
    combined_df = pd.concat([facility_df, sumo1b_df, pd.DataFrame({'Error': error})], axis=1)
    
    print(f"After combining - combined_df shape: {combined_df.shape}")
    print(f"Number of NaN in combined_df: {combined_df.isna().sum().sum()}")

    # Drop any rows that have NaN in any column
    combined_df = combined_df.dropna()
    
    print(f"After dropping all NaN - combined_df shape: {combined_df.shape}")
    print(f"Number of NaN values after dropping: {combined_df.isna().sum().sum()}")

    # Calculate error statistics
    Error_data_mean = combined_df['Error'].mean()
    Error_data_std = combined_df['Error'].std()

    # Save original combined_df before normalization
    original_combined_df = combined_df.copy()

    # Normalize combined_df
    combined_df, normalization_stats = normalize_dataframe(combined_df)

    # Prepare date_time
    combined_df = combined_df.reset_index()
    date_time = pd.to_datetime(combined_df.pop(combined_df.columns[0]), format='%Y-%m-%d')

    # Apply data type specific transformations
    if data_type == 'forecast':
        # Shift all sumo columns by 1 for forecasting
        sumo_cols_to_shift = [col for col in combined_df.columns if col.startswith('sumo_')]
        combined_df[sumo_cols_to_shift] = combined_df[sumo_cols_to_shift].shift(-1)
        combined_df = combined_df.dropna()
    elif data_type == 'correction':
        # Keep only error and sumo columns for correction
        sumo_cols = [col for col in combined_df.columns if col.startswith('sumo_')]
        combined_df = combined_df[['Error'] + sumo_cols]
    elif data_type == 'raw':
        pass  # Return data as is

    return combined_df, date_time, Error_data_mean, Error_data_std, normalization_stats, timeseries_dataframe, timeseries_normalization_stats, original_combined_df, original_timeseries_dataframe

def save_processed_data(combined_df, date_time, Error_data_mean, Error_data_std, normalization_stats, timeseries_dataframe, timeseries_normalization_stats, original_combined_df, original_timeseries_dataframe):
    """Save processed data to output directory"""
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data (normalized versions)
    combined_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    date_time.to_csv(os.path.join(output_dir, 'date_time.csv'), index=False)
    timeseries_dataframe.to_csv(os.path.join(output_dir, 'timeseries_data.csv'), index=False)

    # Save original (non-normalized) versions
    original_combined_df.to_csv(os.path.join(output_dir, 'original_processed_data.csv'), index=False)
    original_timeseries_dataframe.to_csv(os.path.join(output_dir, 'original_timeseries_data.csv'), index=False)

    # Save combined_df statistics
    combined_stats_data = []
    for col, stats in normalization_stats.items():
        combined_stats_data.append({
            'column': col,
            'mean': stats['mean'],
            'std': stats['std']
        })
    
    # Add error statistics to combined stats
    combined_stats_data.append({
        'column': 'Error',
        'mean': Error_data_mean,
        'std': Error_data_std
    })
    
    # Save combined_df mean and std separately
    combined_mean_df = pd.DataFrame([{'mean': Error_data_mean}])
    combined_std_df = pd.DataFrame([{'std': Error_data_std}])
    combined_stats_df = pd.DataFrame(combined_stats_data)
    
    combined_mean_df.to_csv(os.path.join(output_dir, 'combined_df_mean.csv'), index=False)
    combined_std_df.to_csv(os.path.join(output_dir, 'combined_df_std.csv'), index=False)
    combined_stats_df.to_csv(os.path.join(output_dir, 'combined_df_normalization_stats.csv'), index=False)

    # Save timeseries statistics
    timeseries_stats_data = []
    for col, stats in timeseries_normalization_stats.items():
        timeseries_stats_data.append({
            'column': col,
            'mean': stats['mean'],
            'std': stats['std']
        })
    
    # Calculate and save timeseries mean and std separately
    timeseries_mean = timeseries_dataframe.mean().mean()  # Overall mean
    timeseries_std = timeseries_dataframe.std().mean()    # Overall std
    
    timeseries_mean_df = pd.DataFrame([{'mean': timeseries_mean}])
    timeseries_std_df = pd.DataFrame([{'std': timeseries_std}])
    timeseries_stats_df = pd.DataFrame(timeseries_stats_data)
    
    timeseries_mean_df.to_csv(os.path.join(output_dir, 'timeseries_mean.csv'), index=False)
    timeseries_std_df.to_csv(os.path.join(output_dir, 'timeseries_std.csv'), index=False)
    timeseries_stats_df.to_csv(os.path.join(output_dir, 'timeseries_normalization_stats.csv'), index=False)

    print(f"Data saved to {output_dir}/")
    print("Saved files:")
    print("1. processed_data.csv (normalized)")
    print("2. original_processed_data.csv")
    print("3. timeseries_data.csv (normalized)")
    print("4. original_timeseries_data.csv")
    print("5. combined_df_mean.csv")
    print("6. combined_df_std.csv")
    print("7. combined_df_normalization_stats.csv")
    print("8. timeseries_mean.csv")
    print("9. timeseries_std.csv")
    print("10. timeseries_normalization_stats.csv")

if __name__ == "__main__":
    # Example usage
    combined_df, date_time, Error_data_mean, Error_data_std, normalization_stats, timeseries_dataframe, timeseries_normalization_stats, original_combined_df, original_timeseries_dataframe = process_data()
    save_processed_data(combined_df, date_time, Error_data_mean, Error_data_std, normalization_stats, timeseries_dataframe, timeseries_normalization_stats, original_combined_df, original_timeseries_dataframe)