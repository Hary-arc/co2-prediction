import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load data from a CSV or Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean_data(data):
    """
    Clean the data by handling missing values, removing duplicates, etc.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For numeric columns, impute with median
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    # For categorical columns, impute with most frequent value
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def feature_engineering(data):
    """
    Create new features or transform existing ones.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        Data with engineered features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If the dataset contains both CO2 emissions and population data,
    # we can calculate per capita emissions if not already present
    if 'co2_ttl' in df.columns and 'pop' in df.columns and 'co2_per_cap' not in df.columns:
        df['co2_per_cap'] = df['co2_ttl'] / df['pop']
    
    # If GDP and CO2 data are available, calculate emissions intensity (CO2 per unit of GDP)
    if 'co2_ttl' in df.columns and 'gdp' in df.columns and 'co2_per_gdp' not in df.columns:
        df['co2_per_gdp'] = df['co2_ttl'] / df['gdp'] * 1000000  # per million currency units
    
    # If urban population and total population are available, calculate urban population percentage
    if 'urb_pop' in df.columns and 'pop' in df.columns and 'urb_pop_perc' not in df.columns:
        df['urb_pop_perc'] = (df['urb_pop'] / df['pop']) * 100
    
    return df

def normalize_data(data, method='standard'):
    """
    Normalize the data using either standardization or min-max scaling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    method : str, optional
        Normalization method - 'standard' or 'minmax'
        
    Returns:
    --------
    pandas.DataFrame
        Normalized data
    tuple
        Scaler object for inverse transformation
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create appropriate scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization method. Use 'standard' or 'minmax'.")
    
    # Apply scaling
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

def prepare_time_series_data(data, target_col, feature_cols, time_col=None, lag=3):
    """
    Prepare time series data with lagged features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    target_col : str
        Name of the target column
    feature_cols : list
        List of feature column names
    time_col : str, optional
        Name of the time column
    lag : int, optional
        Number of lagged time periods to include
        
    Returns:
    --------
    pandas.DataFrame
        Data with lagged features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If time column is provided, sort by it
    if time_col:
        df = df.sort_values(by=time_col)
    
    # Create lagged features
    for col in feature_cols:
        for i in range(1, lag + 1):
            df[f"{col}_lag_{i}"] = df[col].shift(i)
    
    # Drop rows with NaN due to lagging
    df = df.dropna()
    
    return df
