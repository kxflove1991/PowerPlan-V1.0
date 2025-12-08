import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Optional, List, Dict, Union, Tuple
from src.utils import setup_logger

logger = setup_logger("DataProcessor")

def generate_kmeans_typical_days(ws_df: pd.DataFrame, n_clusters: int = 1) -> pd.DataFrame:
    """
    Apply K-means clustering to find typical days per month.
    
    Args:
        ws_df: DataFrame containing WindPower and SolarPower columns
        n_clusters: Number of clusters (typical days) per month
        
    Returns:
        DataFrame with typical days data and 'Weight' column
    """
    typical_days = []
    
    # Ensure Month and Hour columns exist
    df = ws_df.copy()
    if 'Month' not in df.columns:
        df['Month'] = df.index.month
    if 'Hour' not in df.columns:
        df['Hour'] = df.index.hour
        
    for month in range(1, 13):
        # Filter data for the month
        month_data = df[df['Month'] == month]
        
        if month_data.empty:
            continue

        # Pivot to shape (N_days, 24)
        month_data = month_data.copy()
        month_data['Day'] = month_data.index.date
        
        # Check if we have enough data
        if len(month_data) < 24:
             logger.warning(f"Not enough data for month {month}")
             continue

        # Pivot for Wind and Solar
        try:
            wind_pivot = month_data.pivot(index='Day', columns='Hour', values='WindPower')
            solar_pivot = month_data.pivot(index='Day', columns='Hour', values='SolarPower')
        except ValueError:
            # Handle duplicate entries if any
            logger.warning(f"Duplicate time entries found in month {month}, aggregating mean.")
            month_data_grouped = month_data.groupby(['Day', 'Hour']).mean().reset_index()
            wind_pivot = month_data_grouped.pivot(index='Day', columns='Hour', values='WindPower')
            solar_pivot = month_data_grouped.pivot(index='Day', columns='Hour', values='SolarPower')

        # Drop incomplete days
        wind_pivot = wind_pivot.dropna()
        solar_pivot = solar_pivot.dropna()
        
        # Intersection of days to ensure alignment
        common_days = wind_pivot.index.intersection(solar_pivot.index)
        wind_pivot = wind_pivot.loc[common_days]
        solar_pivot = solar_pivot.loc[common_days]
        
        if len(common_days) == 0:
            logger.warning(f"No valid complete days for month {month}")
            continue
            
        # Feature vector: Wind + Solar (Size 48 per day)
        features = np.hstack([wind_pivot.values, solar_pivot.values])
        
        # K-Means clustering
        actual_k = min(n_clusters, len(features))
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # Calculate weights (number of days in each cluster)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        weights_map = dict(zip(unique, counts))
        
        for i, center in enumerate(kmeans.cluster_centers_):
            typical_wind = center[:24]
            typical_solar = center[24:]
            weight = weights_map.get(i, 0)
            
            # Construct a time index for this typical day
            # We assign it to day i+1 of the month for reference (Synthetic Date)
            
            df_day = pd.DataFrame({
                'Month': month,
                'Day': i + 1, # 1, 2, 3...
                'Hour': range(24),
                'WindPower': typical_wind,
                'SolarPower': typical_solar,
                'Weight': weight # Same weight for all hours in the day
            })
            typical_days.append(df_day)
        
    if not typical_days:
        return pd.DataFrame(columns=['Month', 'Day', 'Hour', 'WindPower', 'SolarPower', 'Weight'])
        
    return pd.concat(typical_days, ignore_index=True)

def _load_and_clean_ws(wind_solar_path: str) -> Optional[pd.DataFrame]:
    """Load and clean wind/solar data."""
    if not os.path.exists(wind_solar_path):
        logger.error(f"File not found: {wind_solar_path}")
        return None
        
    try:
        ws_df = pd.read_csv(wind_solar_path)
    except Exception as e:
        logger.error(f"Failed to read CSV {wind_solar_path}: {e}")
        return None
    
    # Build datetime index
    try:
        if 'Time' in ws_df.columns:
            ws_df['Time'] = ws_df['Time'].astype(str).apply(lambda x: x.split(' ')[-1])

        if 'Date' in ws_df.columns and 'Time' in ws_df.columns:
            ws_df['Datetime'] = pd.to_datetime(ws_df['Date'] + ' ' + ws_df['Time'])
        else:
            # Fallback to default 2022 hourly index
            logger.info("Date/Time columns missing, using default 2022 hourly index.")
            ws_df['Datetime'] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")
    except Exception as e:
        logger.warning(f"Time parsing failed ({e}), using default 8760 index.")
        ws_df['Datetime'] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")
        
    ws_df.set_index('Datetime', inplace=True)
    
    # Clean numeric data
    for col in ['WindPower', 'SolarPower']:
        if col in ws_df.columns:
            ws_df[col] = pd.to_numeric(ws_df[col], errors='coerce').fillna(0)
        else:
            logger.warning(f"Column '{col}' missing in {wind_solar_path}, filling with 0.")
            ws_df[col] = 0.0
    
    return ws_df

def _process_load_data(load_path: str, target_index: pd.DatetimeIndex) -> Optional[np.ndarray]:
    """Map monthly typical load profile to target index."""
    if not os.path.exists(load_path):
        logger.error(f"File not found: {load_path}")
        return None

    try:
        load_raw = pd.read_csv(load_path, index_col=0)
    except Exception as e:
        logger.error(f"Failed to read load data {load_path}: {e}")
        return None
        
    load_raw.columns = [str(c).strip() for c in load_raw.columns]
    
    full_year_load = []
    month_columns = load_raw.columns
    
    # Iterate through target index (could be 8760 or typical days)
    for idx in target_index:
        month = idx.month
        hour = idx.hour
        
        # Column selection (months are 1-based, columns are 0-based index)
        col_idx = month - 1
        if col_idx >= len(month_columns):
            col_idx = 0 # Fallback
            
        try:
            val = load_raw.iloc[hour, col_idx]
        except IndexError:
            logger.warning(f"Index error for load data at Month {month}, Hour {hour}. Using fallback.")
            val = 2400000.0 # Fallback (should be avoided if possible)
            
        full_year_load.append(val)
        
    load_mw = np.array(full_year_load) / 1000.0
    return load_mw

def process_input_data(wind_solar_path: str, load_path: str, output_path: str = 'clean_input_data.csv') -> Optional[pd.DataFrame]:
    """
    Process real data: Use raw 8760 Wind/Solar data + Mapped Load data.
    Used for Validation.
    """
    logger.info("Processing validation data (Real 8760)...")
    
    ws_df = _load_and_clean_ws(wind_solar_path)
    if ws_df is None: return None
    
    load_mw = _process_load_data(load_path, ws_df.index)
    if load_mw is None: return None
    
    clean_df = pd.DataFrame({
        'wind_p_max_pu': ws_df['WindPower'],
        'solar_p_max_pu': ws_df['SolarPower'],
        'load_mw': load_mw
    }, index=ws_df.index)
    
    # Clip values
    clean_df['wind_p_max_pu'] = clean_df['wind_p_max_pu'].clip(0, 1)
    clean_df['solar_p_max_pu'] = clean_df['solar_p_max_pu'].clip(0, 1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    clean_df.to_csv(output_path)
    logger.info(f"Validation data saved to {output_path}")
    return clean_df

def process_input_data_typical(wind_solar_path: str, load_path: str, output_path: str = 'clean_input_data_typical.csv', n_clusters: int = 1) -> Optional[pd.DataFrame]:
    """
    Process optimization data: Use K-means to generate typical days.
    """
    logger.info(f"Processing optimization data (K-means Typical, k={n_clusters})...")
    
    ws_df = _load_and_clean_ws(wind_solar_path)
    if ws_df is None: return None
    
    # --- K-means Clustering ---
    logger.info("Running K-means clustering...")
    typical_days_df = generate_kmeans_typical_days(ws_df, n_clusters=n_clusters)
    
    if typical_days_df.empty:
        logger.error("Failed to generate typical days.")
        return None
        
    # --- Construct Output Data ---
    timestamps = []
    
    for _, row in typical_days_df.iterrows():
        # Construct timestamp: 2022-Month-Day Hour:00
        # Using a fixed year 2022 for typical days
        try:
            ts = pd.Timestamp(year=2022, month=int(row['Month']), day=int(row['Day']), hour=int(row['Hour']))
        except ValueError:
            # Handle leap year or invalid date if necessary (simplified here)
            ts = pd.Timestamp(year=2022, month=1, day=1, hour=0) 
        timestamps.append(ts)
        
    typical_days_df.index = pd.DatetimeIndex(timestamps)
    typical_days_df = typical_days_df.sort_index()
    
    # Process Load (Map typical hours to load)
    load_mw = _process_load_data(load_path, typical_days_df.index)
    
    clean_df = pd.DataFrame({
        'wind_p_max_pu': typical_days_df['WindPower'],
        'solar_p_max_pu': typical_days_df['SolarPower'],
        'load_mw': load_mw,
        'weight': typical_days_df['Weight']
    }, index=typical_days_df.index)
    
    # Clip values
    clean_df['wind_p_max_pu'] = clean_df['wind_p_max_pu'].clip(0, 1)
    clean_df['solar_p_max_pu'] = clean_df['solar_p_max_pu'].clip(0, 1)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    clean_df.to_csv(output_path)
    logger.info(f"Typical days data saved to {output_path} (Rows: {len(clean_df)})")
    return clean_df

if __name__ == "__main__":
    # Test run
    print("Starting Data Processing...")
    process_input_data('data/Wind_Solar_Power.csv', 'data/load_data.csv', 'results/validation_input.csv')
    process_input_data_typical('data/Wind_Solar_Power.csv', 'data/load_data.csv', 'results/optimization_input.csv')
