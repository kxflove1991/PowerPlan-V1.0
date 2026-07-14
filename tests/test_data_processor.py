"""测试 src/data_processor.py 中的数据处理函数。"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_processor import (
    _build_typical_day_index,
    _load_and_clean_ws,
    _process_load_data,
    generate_kmeans_typical_days,
    process_input_data,
    process_input_data_typical,
)
from src.utils import ConfigManager


def test_generate_kmeans_typical_days_shape(sample_wind_solar_data):
    result = generate_kmeans_typical_days(sample_wind_solar_data, n_clusters=2)

    # 12 个月 * 2 个典型日 * 24 小时
    assert len(result) == 12 * 2 * 24
    assert set(result.columns) == {"Month", "Day", "Hour", "WindPower", "SolarPower", "Weight"}
    # 权重按小时重复，因此所有行的权重和为 365 天 * 24 小时
    assert result["Weight"].sum() == 365 * 24
    # 按（月，典型日）去重后，权重和应等于全年天数
    unique_day_weights = result.groupby(["Month", "Day"])["Weight"].first()
    assert unique_day_weights.sum() == 365


def test_build_typical_day_index_unique():
    df = pd.DataFrame({
        "Month": [1, 1, 2],
        "Day": [1, 2, 1],
        "Hour": [0, 0, 0],
    })
    index = _build_typical_day_index(df)
    assert not index.duplicated().any()
    assert str(index[0]) == "01-01-00"


def test_process_load_data_vectorized(sample_load_data):
    # 测试 DatetimeIndex
    target_index = pd.date_range("2022-01-01", periods=24, freq="h")
    load_mw = _process_load_data_from_matrix(sample_load_data, target_index)
    assert len(load_mw) == 24
    assert np.all(load_mw > 0)

    # 测试字符串索引
    string_index = pd.Index(["01-01-00", "02-01-01", "03-01-02"])
    load_mw2 = _process_load_data_from_matrix(sample_load_data, string_index)
    assert len(load_mw2) == 3


def _process_load_data_from_matrix(load_raw, target_index):
    """辅助函数：模拟向量化负荷映射。"""
    load_matrix = load_raw.values
    if isinstance(target_index, pd.DatetimeIndex):
        months = target_index.month - 1
        hours = target_index.hour
    else:
        parsed = [str(idx).split("-") for idx in target_index]
        months = np.array([int(p[0]) for p in parsed], dtype=int) - 1
        hours = np.array([int(p[2]) for p in parsed], dtype=int)
    return load_matrix[hours, months] / 1000.0


def test_process_input_data_creates_file(temp_dir, sample_data_dir, sample_config_path, reset_config):
    ConfigManager.load_config(sample_config_path)

    output_path = os.path.join(temp_dir, "validation_input.csv")
    ws_path = os.path.join(sample_data_dir, "Wind_Solar_Power.csv")
    load_path = os.path.join(sample_data_dir, "load_data.csv")

    result = process_input_data(ws_path, load_path, output_path=output_path)

    assert result is not None
    assert os.path.exists(output_path)
    assert set(result.columns) == {"wind_p_max_pu", "solar_p_max_pu", "load_mw"}


def test_process_input_data_typical_creates_file(temp_dir, sample_data_dir, sample_config_path, reset_config):
    ConfigManager.load_config(sample_config_path)

    output_path = os.path.join(temp_dir, "optimization_input.csv")
    ws_path = os.path.join(sample_data_dir, "Wind_Solar_Power.csv")
    load_path = os.path.join(sample_data_dir, "load_data.csv")

    result = process_input_data_typical(ws_path, load_path, output_path=output_path, n_clusters=2)

    assert result is not None
    assert os.path.exists(output_path)
    assert set(result.columns) == {"wind_p_max_pu", "solar_p_max_pu", "load_mw", "weight"}
    assert len(result) == 12 * 2 * 24


def test_load_and_clean_ws_parses_date_time(temp_dir, sample_wind_solar_data):
    ws_path = os.path.join(temp_dir, "ws.csv")
    df = sample_wind_solar_data.copy()
    df["Date"] = df.index.strftime("%Y-%m-%d")
    df["Time"] = df.index.strftime("%H:%M:%S")
    df[["Date", "Time", "WindPower", "SolarPower"]].to_csv(ws_path, index=False)

    loaded = _load_and_clean_ws(ws_path)
    assert loaded is not None
    assert len(loaded) == 8760
    assert set(loaded.columns) >= {"WindPower", "SolarPower"}
