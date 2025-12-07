import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

def generate_kmeans_typical_days(ws_df, n_clusters=1):
    """
    Apply K-means clustering to find typical days per month.
    Returns:
        typical_days_df: DataFrame with typical days data and 'Weight' column
    """
    typical_days = []
    
    # Ensure Month and Hour columns exist
    # Use a copy to avoid SettingWithCopyWarning
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
             print(f"Warning: Not enough data for month {month}")
             continue

        # Pivot for Wind
        try:
            wind_pivot = month_data.pivot(index='Day', columns='Hour', values='WindPower')
            solar_pivot = month_data.pivot(index='Day', columns='Hour', values='SolarPower')
        except ValueError as e:
            # Handle duplicate entries if any
            print(f"Duplicate time entries found in month {month}, aggregating mean.")
            month_data_grouped = month_data.groupby(['Day', 'Hour']).mean().reset_index()
            wind_pivot = month_data_grouped.pivot(index='Day', columns='Hour', values='WindPower')
            solar_pivot = month_data_grouped.pivot(index='Day', columns='Hour', values='SolarPower')

        # Drop incomplete days
        wind_pivot = wind_pivot.dropna()
        solar_pivot = solar_pivot.dropna()
        
        # Intersection of days
        common_days = wind_pivot.index.intersection(solar_pivot.index)
        wind_pivot = wind_pivot.loc[common_days]
        solar_pivot = solar_pivot.loc[common_days]
        
        if len(common_days) == 0:
            print(f"Warning: No valid complete days for month {month}")
            continue
            
        # Feature vector: Wind + Solar (Size 48)
        features = np.hstack([wind_pivot.values, solar_pivot.values])
        
        # K-Means clustering
        # Ensure we don't ask for more clusters than samples
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
            # We assign it to day i+1 of the month for reference
            # Note: This is a synthetic date
            
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

def _load_and_clean_ws(wind_solar_path):
    if not os.path.exists(wind_solar_path):
        print(f"错误: 找不到文件 {wind_solar_path}")
        return None
        
    ws_df = pd.read_csv(wind_solar_path)
    
    # 构建时间索引
    try:
        if 'Time' in ws_df.columns:
            ws_df['Time'] = ws_df['Time'].astype(str).apply(lambda x: x.split(' ')[-1])

        if 'Date' in ws_df.columns and 'Time' in ws_df.columns:
            ws_df['Datetime'] = pd.to_datetime(ws_df['Date'] + ' ' + ws_df['Time'])
        else:
            ws_df['Datetime'] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")
    except Exception as e:
        print(f"时间列解析失败，尝试使用默认 8760 索引: {e}")
        ws_df['Datetime'] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")
        
    ws_df.set_index('Datetime', inplace=True)
    
    # 清洗数据
    ws_df['WindPower'] = pd.to_numeric(ws_df['WindPower'], errors='coerce').fillna(0)
    ws_df['SolarPower'] = pd.to_numeric(ws_df['SolarPower'], errors='coerce').fillna(0)
    
    return ws_df

def _process_load_data(load_path, target_index):
    if not os.path.exists(load_path):
        print(f"错误: 找不到文件 {load_path}")
        return None

    load_raw = pd.read_csv(load_path, index_col=0)
    load_raw.columns = [str(c).strip() for c in load_raw.columns]
    
    full_year_load = []
    month_columns = load_raw.columns
    
    # 遍历 8760 小时
    for idx in target_index:
        month = idx.month
        hour = idx.hour
        
        if month <= len(month_columns):
            col_name = month_columns[month-1]
        else:
            col_name = month_columns[0]
            
        try:
            val = load_raw.iloc[hour, month-1]
        except:
            val = 2400000 
            
        full_year_load.append(val)
        
    load_mw = np.array(full_year_load) / 1000.0
    return load_mw

def process_input_data(wind_solar_path, load_path, output_path='clean_input_data.csv'):
    """
    处理真实数据：直接使用原始 Wind/Solar 8760 数据 + 映射负荷数据。
    用于：校验 (Validation)
    """
    print(f"正在处理输入数据 (真实 8760)...")
    
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

    clean_df.to_csv(output_path)
    print(f"处理完成! 真实 8760 数据已保存至 {output_path}")
    return clean_df

def process_input_data_typical(wind_solar_path, load_path, output_path='clean_input_data_typical.csv', n_clusters=1):
    """
    处理典型日数据：使用 K-means 生成月典型日。
    直接输出典型日数据（不平铺为 8760），由优化模型根据权重处理。
    """
    print(f"正在处理输入数据 (K-means 典型日, k={n_clusters})...")
    
    ws_df = _load_and_clean_ws(wind_solar_path)
    if ws_df is None: return None
    
    # --- K-means 生成典型日 ---
    print("正在进行 K-means 聚类生成月典型日...")
    typical_days_df = generate_kmeans_typical_days(ws_df, n_clusters=n_clusters)
    
    if typical_days_df.empty:
        print("错误: 无法生成典型日数据")
        return None
        
    # --- 构造输出数据 ---
    # 我们需要为这些典型日构造一个 datetime 索引，以便 xarray/pypsa 识别
    # 我们使用 2022 年作为基准年，将典型日映射到每月的前 k 天
    
    timestamps = []
    
    for _, row in typical_days_df.iterrows():
        # Construct timestamp: 2022-Month-Day Hour:00
        ts = pd.Timestamp(year=2022, month=int(row['Month']), day=int(row['Day']), hour=int(row['Hour']))
        timestamps.append(ts)
        
    typical_days_df.index = pd.DatetimeIndex(timestamps)
    typical_days_df = typical_days_df.sort_index()
    
    # Process Load (Map typical hours to load)
    # 简单的做法：取该月该小时的平均负荷，或者直接取对应时刻的负荷
    # 这里我们复用 _process_load_data，但只取对应的时间点
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
    
    clean_df.to_csv(output_path)
    print(f"处理完成! 典型日数据已保存至 {output_path} (Rows: {len(clean_df)})")
    return clean_df

if __name__ == "__main__":
    # Test run
    print("开始执行 K-means 数据处理流程...")
    # 1. Validation Data (Real)
    process_input_data('Wind_Solar_Power.csv', 'load_data.csv', 'validation_input.csv')
    # 2. Optimization Data (K-means Typical)
    process_input_data_typical('Wind_Solar_Power.csv', 'load_data.csv', 'optimization_input.csv')