"""
input_validation.py
-------------------
输入数据校验模块，用于在数据进入模型前检查其完整性、一致性和合理性。
"""

from typing import Set

import pandas as pd


class InputValidationError(ValueError):
    """输入数据校验失败时抛出的异常。"""


def validate_wind_solar_data(df: pd.DataFrame, expected_hours: int = 8760) -> None:
    """
    校验风光出力原始数据。

    Args:
        df: 包含 WindPower 和 SolarPower 列的 DataFrame
        expected_hours: 期望的时间序列长度，默认 8760

    Raises:
        InputValidationError: 当数据不符合要求时
    """
    required_cols: Set[str] = {"WindPower", "SolarPower"}
    missing = required_cols - set(df.columns)
    if missing:
        raise InputValidationError(f"风光数据缺少必要列: {missing}")

    if len(df) != expected_hours:
        raise InputValidationError(f"风光数据期望 {expected_hours} 行，实际 {len(df)} 行")

    if df.index.duplicated().any():
        raise InputValidationError("风光数据存在重复时间戳")

    if not df.index.is_monotonic_increasing:
        raise InputValidationError("风光数据时间戳未按升序排列")

    if not df["WindPower"].between(0, 1).all():
        raise InputValidationError("WindPower 取值应在 [0, 1] 之间（标幺值）")

    if not df["SolarPower"].between(0, 1).all():
        raise InputValidationError("SolarPower 取值应在 [0, 1] 之间（标幺值）")


def validate_load_data(df: pd.DataFrame) -> None:
    """
    校验负荷数据。

    Args:
        df: 负荷数据 DataFrame，形状应为 (24, 12)

    Raises:
        InputValidationError: 当数据不符合要求时
    """
    if df.empty:
        raise InputValidationError("负荷数据为空")

    if df.shape[0] != 24:
        raise InputValidationError(f"负荷数据期望 24 行（每小时一行），实际 {df.shape[0]} 行")

    if df.shape[1] != 12:
        raise InputValidationError(f"负荷数据期望 12 列（每月一列），实际 {df.shape[1]} 列")

    if (df < 0).any().any():
        raise InputValidationError("负荷存在负值")


def validate_optimization_input(df: pd.DataFrame) -> None:
    """
    校验优化/校验阶段的输入数据。

    Args:
        df: 包含 wind_p_max_pu、solar_p_max_pu、load_mw 列的 DataFrame

    Raises:
        InputValidationError: 当数据不符合要求时
    """
    required_cols: Set[str] = {"wind_p_max_pu", "solar_p_max_pu", "load_mw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise InputValidationError(f"优化输入缺少必要列: {missing}")

    if df.index.duplicated().any():
        raise InputValidationError("优化输入存在重复时间戳")

    if (df["load_mw"] < 0).any():
        raise InputValidationError("优化输入中负荷存在负值")

    if not df["wind_p_max_pu"].between(0, 1).all():
        raise InputValidationError("优化输入中 wind_p_max_pu 超出 [0,1]")

    if not df["solar_p_max_pu"].between(0, 1).all():
        raise InputValidationError("优化输入中 solar_p_max_pu 超出 [0,1]")
