"""测试 src/input_validation.py 中的校验函数。"""

import numpy as np
import pandas as pd
import pytest

from src.input_validation import (
    InputValidationError,
    validate_load_data,
    validate_optimization_input,
    validate_wind_solar_data,
)


def test_validate_wind_solar_data_ok():
    index = pd.date_range("2022-01-01", periods=8760, freq="h")
    df = pd.DataFrame({
        "WindPower": np.random.rand(8760),
        "SolarPower": np.random.rand(8760),
    }, index=index)
    validate_wind_solar_data(df)  # should not raise


def test_validate_wind_solar_data_missing_column():
    df = pd.DataFrame({"WindPower": np.random.rand(8760)})
    with pytest.raises(InputValidationError, match="缺少必要列"):
        validate_wind_solar_data(df)


def test_validate_wind_solar_data_wrong_length():
    df = pd.DataFrame({
        "WindPower": np.random.rand(100),
        "SolarPower": np.random.rand(100),
    })
    with pytest.raises(InputValidationError, match="期望 8760 行"):
        validate_wind_solar_data(df)


def test_validate_wind_solar_data_out_of_range():
    df = pd.DataFrame({
        "WindPower": np.full(8760, 1.5),
        "SolarPower": np.full(8760, 0.5),
    })
    with pytest.raises(InputValidationError, match="WindPower"):
        validate_wind_solar_data(df)


def test_validate_load_data_ok():
    df = pd.DataFrame({m: np.random.randint(1_000_000, 2_000_000, size=24) for m in range(1, 13)})
    validate_load_data(df)  # should not raise


def test_validate_load_data_negative():
    df = pd.DataFrame({m: np.full(24, -100) for m in range(1, 13)})
    with pytest.raises(InputValidationError, match="负荷存在负值"):
        validate_load_data(df)


def test_validate_optimization_input_ok():
    df = pd.DataFrame({
        "wind_p_max_pu": np.random.rand(24),
        "solar_p_max_pu": np.random.rand(24),
        "load_mw": np.random.rand(24) * 1000,
    })
    validate_optimization_input(df)  # should not raise


def test_validate_optimization_input_missing_column():
    df = pd.DataFrame({"wind_p_max_pu": np.random.rand(24)})
    with pytest.raises(InputValidationError, match="缺少必要列"):
        validate_optimization_input(df)
