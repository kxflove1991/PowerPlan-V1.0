"""
conftest.py
-----------
pytest 共享 fixtures。
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.utils import ConfigManager


@pytest.fixture
def reset_config():
    """在每个测试前重置 ConfigManager。"""
    ConfigManager.reset()
    yield
    ConfigManager.reset()


@pytest.fixture
def temp_dir():
    """创建临时目录，测试结束后自动清理。"""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_wind_solar_data():
    """生成 8760 小时风光出力样本数据。"""
    np.random.seed(42)
    index = pd.date_range("2022-01-01", periods=8760, freq="h")
    wind = np.clip(np.random.rand(8760) * 0.8 + 0.1, 0, 1)
    solar = np.clip(np.where(
        (index.hour >= 6) & (index.hour <= 18),
        np.random.rand(8760) * 0.9,
        0
    ), 0, 1)
    return pd.DataFrame({
        "WindPower": wind,
        "SolarPower": solar
    }, index=index)


@pytest.fixture
def sample_load_data():
    """生成 24x12 负荷样本数据（单位：kW）。"""
    np.random.seed(42)
    hours = range(24)
    months = range(1, 13)
    data = {m: np.random.randint(2_000_000, 3_000_000, size=24) for m in months}
    return pd.DataFrame(data, index=hours)


@pytest.fixture
def sample_config_path(temp_dir):
    """生成一个最小可用配置文件。"""
    config_path = os.path.join(temp_dir, "config.yaml")
    config_content = """
settings:
  typical_days_per_month: 1
  allow_suboptimal_export: false
  strict_validation: true

financial:
  discount_rate: 0.05
  lifetime_years: 20

penalties:
  load_shedding: 100000.0
  curtailment: 100.0

costs:
  wind:
    capex: 3300.0
    opex: 30.0
  solar:
    capex: 2900.0
    opex: 27.0
  thermal:
    capex: 4800.0
    opex: 10.0
    fuel_cost: 0.185
  storage:
    power_capex: 800.0
    energy_capex: 750.0
    opex: 60.0
    efficiency: 0.927

constraints:
  penalties:
    enable_curtailment_penalty: true
    curtailment: 100.0

  transmission:
    voltage_level: 800
    capacity: 8000
    min_utilization_hours: 4000
    min_re_share: 0.50
    max_curtailment_rate: 0.10
    max_load_shedding_rate: 0.0

  storage:
    min_duration: 4.0
    max_duration: 4.0
    min_capacity_ratio: 0.15
    max_capacity_ratio: 0.20
    power_capacity_min: 0
    power_capacity_max: .inf
    energy_capacity_min: 0
    energy_capacity_max: .inf

  thermal:
    min_load_rate: 0.20
    unit_sizes: [660, 1000]
    capacity_min: 2640
    capacity_max: 2640

  renewable:
    wind:
      capacity_min: 0
      capacity_max: .inf
    solar:
      capacity_min: 0
      capacity_max: .inf

solver:
  name: highs
  options:
    threads: 4
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return config_path


@pytest.fixture
def sample_data_dir(temp_dir, sample_wind_solar_data, sample_load_data):
    """创建包含样本输入数据的临时目录。"""
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    ws_path = os.path.join(data_dir, "Wind_Solar_Power.csv")
    # 保存为 Date, Time, WindPower, SolarPower 格式
    df = sample_wind_solar_data.copy()
    df["Date"] = df.index.strftime("%Y-%m-%d")
    df["Time"] = df.index.strftime("%H:%M:%S")
    df[["Date", "Time", "WindPower", "SolarPower"]].to_csv(ws_path, index=False)

    load_path = os.path.join(data_dir, "load_data.csv")
    sample_load_data.to_csv(load_path)

    return data_dir
