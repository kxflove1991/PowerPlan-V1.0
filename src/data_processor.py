"""
data_processor.py
-----------------
输入数据预处理模块：
- 读取风光出力原始数据并清洗
- 使用 K-Means 聚类生成每月典型日
- 将月负荷曲线映射到目标时间索引
- 输出优化和校验所需的 CSV 文件
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.input_validation import (
    validate_load_data,
    validate_optimization_input,
    validate_wind_solar_data,
)
from src.utils import setup_logger

logger = setup_logger("DataProcessor")


def generate_kmeans_typical_days(ws_df: pd.DataFrame, n_clusters: int = 1) -> pd.DataFrame:
    """
    按月对风光出力进行 K-Means 聚类，生成典型日。

    Args:
        ws_df: 包含 WindPower 和 SolarPower 列的 DataFrame，索引为时间戳
        n_clusters: 每月典型日数量

    Returns:
        包含典型日出力及权重的 DataFrame
    """
    typical_days = []

    df = ws_df.copy()
    if "Month" not in df.columns:
        df["Month"] = df.index.month
    if "Hour" not in df.columns:
        df["Hour"] = df.index.hour

    for month in range(1, 13):
        month_data = df[df["Month"] == month]

        if month_data.empty:
            continue

        month_data = month_data.copy()
        month_data["Day"] = month_data.index.date

        if len(month_data) < 24:
            logger.warning(f"Not enough data for month {month}")
            continue

        try:
            wind_pivot = month_data.pivot(index="Day", columns="Hour", values="WindPower")
            solar_pivot = month_data.pivot(index="Day", columns="Hour", values="SolarPower")
        except ValueError:
            logger.warning(f"Duplicate time entries found in month {month}, aggregating mean.")
            month_data_grouped = month_data.groupby(["Day", "Hour"]).mean().reset_index()
            wind_pivot = month_data_grouped.pivot(index="Day", columns="Hour", values="WindPower")
            solar_pivot = month_data_grouped.pivot(index="Day", columns="Hour", values="SolarPower")

        wind_pivot = wind_pivot.dropna()
        solar_pivot = solar_pivot.dropna()

        common_days = wind_pivot.index.intersection(solar_pivot.index)
        wind_pivot = wind_pivot.loc[common_days]
        solar_pivot = solar_pivot.loc[common_days]

        if len(common_days) == 0:
            logger.warning(f"No valid complete days for month {month}")
            continue

        # 特征向量：风 + 光（48 维）
        features = np.hstack([wind_pivot.values, solar_pivot.values])

        # 标准化特征，避免某一能源量纲主导聚类
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        actual_k = min(n_clusters, len(features_scaled))
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)

        # 聚类中心反标准化回原始量纲
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        weights_map = dict(zip(unique, counts))

        for i, center in enumerate(centers):
            typical_wind = center[:24]
            typical_solar = center[24:]
            weight = weights_map.get(i, 0)

            df_day = pd.DataFrame({
                "Month": month,
                "Day": i + 1,
                "Hour": range(24),
                "WindPower": typical_wind,
                "SolarPower": typical_solar,
                "Weight": weight
            })
            typical_days.append(df_day)

    if not typical_days:
        return pd.DataFrame(columns=["Month", "Day", "Hour", "WindPower", "SolarPower", "Weight"])

    return pd.concat(typical_days, ignore_index=True)


def _load_and_clean_ws(wind_solar_path: str) -> Optional[pd.DataFrame]:
    """加载并清洗风光出力数据。"""
    if not os.path.exists(wind_solar_path):
        logger.error(f"File not found: {wind_solar_path}")
        return None

    try:
        ws_df = pd.read_csv(wind_solar_path)
    except Exception as e:
        logger.error(f"Failed to read CSV {wind_solar_path}: {e}")
        return None

    # 构造时间索引
    try:
        if "Time" in ws_df.columns:
            ws_df["Time"] = ws_df["Time"].astype(str).apply(lambda x: x.split(" ")[-1])

        if "Date" in ws_df.columns and "Time" in ws_df.columns:
            ws_df["Datetime"] = pd.to_datetime(ws_df["Date"] + " " + ws_df["Time"])
        else:
            logger.info("Date/Time columns missing, using default 2022 hourly index.")
            ws_df["Datetime"] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")
    except Exception as e:
        logger.warning(f"Time parsing failed ({e}), using default 8760 index.")
        ws_df["Datetime"] = pd.date_range("2022-01-01", periods=len(ws_df), freq="h")

    ws_df.set_index("Datetime", inplace=True)

    # 清洗数值列
    for col in ["WindPower", "SolarPower"]:
        if col in ws_df.columns:
            ws_df[col] = pd.to_numeric(ws_df[col], errors="coerce").fillna(0)
        else:
            logger.warning(f"Column '{col}' missing in {wind_solar_path}, filling with 0.")
            ws_df[col] = 0.0

    validate_wind_solar_data(ws_df)
    return ws_df


def _process_load_data(load_path: str, target_index: pd.Index) -> Optional[np.ndarray]:
    """将月负荷曲线映射到目标时间索引。"""
    if not os.path.exists(load_path):
        logger.error(f"File not found: {load_path}")
        return None

    try:
        load_raw = pd.read_csv(load_path, index_col=0)
    except Exception as e:
        logger.error(f"Failed to read load data {load_path}: {e}")
        return None

    validate_load_data(load_raw)

    load_matrix = load_raw.values  # shape (24, 12)

    # 提取月份和小时（支持 DatetimeIndex 或字符串索引 MM-DD-HH）
    if isinstance(target_index, pd.DatetimeIndex):
        months = target_index.month - 1  # 0-based
        hours = target_index.hour
    else:
        parsed = [str(idx).split("-") for idx in target_index]
        months = np.array([int(p[0]) for p in parsed], dtype=int) - 1
        hours = np.array([int(p[2]) for p in parsed], dtype=int)

    load_mw = load_matrix[hours, months] / 1000.0
    return load_mw


def _build_typical_day_index(typical_days_df: pd.DataFrame) -> pd.Index:
    """为典型日构造无冲突的唯一索引：MM-DD-HH。"""
    labels = (
        typical_days_df["Month"].astype(int).astype(str).str.zfill(2) + "-"
        + typical_days_df["Day"].astype(int).astype(str).str.zfill(2) + "-"
        + typical_days_df["Hour"].astype(int).astype(str).str.zfill(2)
    )
    return pd.Index(labels, name="typical_period")


def process_input_data(
    wind_solar_path: str,
    load_path: str,
    output_path: str = "clean_input_data.csv"
) -> Optional[pd.DataFrame]:
    """处理 8760 小时真实数据，用于校验。"""
    logger.info("Processing validation data (Real 8760)...")

    ws_df = _load_and_clean_ws(wind_solar_path)
    if ws_df is None:
        return None

    load_mw = _process_load_data(load_path, ws_df.index)
    if load_mw is None:
        return None

    clean_df = pd.DataFrame({
        "wind_p_max_pu": ws_df["WindPower"],
        "solar_p_max_pu": ws_df["SolarPower"],
        "load_mw": load_mw
    }, index=ws_df.index)

    clean_df["wind_p_max_pu"] = clean_df["wind_p_max_pu"].clip(0, 1)
    clean_df["solar_p_max_pu"] = clean_df["solar_p_max_pu"].clip(0, 1)

    validate_optimization_input(clean_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    clean_df.to_csv(output_path)
    logger.info(f"Validation data saved to {output_path}")
    return clean_df


def process_input_data_typical(
    wind_solar_path: str,
    load_path: str,
    output_path: str = "clean_input_data_typical.csv",
    n_clusters: int = 1
) -> Optional[pd.DataFrame]:
    """处理典型日数据，用于优化。"""
    logger.info(f"Processing optimization data (K-means Typical, k={n_clusters})...")

    ws_df = _load_and_clean_ws(wind_solar_path)
    if ws_df is None:
        return None

    typical_days_df = generate_kmeans_typical_days(ws_df, n_clusters=n_clusters)

    if typical_days_df.empty:
        logger.error("Failed to generate typical days.")
        return None

    typical_days_df.index = _build_typical_day_index(typical_days_df)
    typical_days_df = typical_days_df.sort_index()

    load_mw = _process_load_data(load_path, typical_days_df.index)
    if load_mw is None:
        return None

    clean_df = pd.DataFrame({
        "wind_p_max_pu": typical_days_df["WindPower"],
        "solar_p_max_pu": typical_days_df["SolarPower"],
        "load_mw": load_mw,
        "weight": typical_days_df["Weight"]
    }, index=typical_days_df.index)

    clean_df["wind_p_max_pu"] = clean_df["wind_p_max_pu"].clip(0, 1)
    clean_df["solar_p_max_pu"] = clean_df["solar_p_max_pu"].clip(0, 1)

    validate_optimization_input(clean_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    clean_df.to_csv(output_path)
    logger.info(f"Typical days data saved to {output_path} (Rows: {len(clean_df)})")
    return clean_df


if __name__ == "__main__":
    print("Starting Data Processing...")
    process_input_data("data/Wind_Solar_Power.csv", "data/load_data.csv", "results/validation_input.csv")
    process_input_data_typical("data/Wind_Solar_Power.csv", "data/load_data.csv", "results/optimization_input.csv")
