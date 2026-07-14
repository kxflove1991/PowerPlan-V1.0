"""
utils.py
--------
通用工具模块，包含日志配置、配置管理、求解器状态判断和 PyPSA 权重获取等工具函数。
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union

import pandas as pd
import pypsa


def setup_logger(name: str = "PowerPlan", level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with standard formatting.

    Args:
        name: Name of the logger
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


class ConfigManager:
    """Singleton-like configuration manager."""
    _config = None

    @classmethod
    def load_config(cls, path: str = 'config/config.yaml') -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            path: Path to config file

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if cls._config is None:
            # Try default path if not loaded
            try:
                return cls.load_config()
            except Exception:
                raise ValueError("Configuration not loaded and default path failed. Call load_config() first.")
        return cls._config

    @classmethod
    def reset(cls) -> None:
        """Reset the cached configuration. Useful for tests."""
        cls._config = None


def get_weight_series(n: pypsa.Network) -> pd.Series:
    """
    统一获取 PyPSA 网络的 snapshot weightings。

    兼容 PyPSA 不同版本下 snapshot_weightings 为 Series 或 DataFrame 的情况。
    """
    sw = n.snapshot_weightings
    if isinstance(sw, pd.DataFrame):
        return sw.iloc[:, 0]
    return sw


def is_solver_optimal(status: Union[str, tuple, Any]) -> bool:
    """
    判断求解器返回状态是否为最优或成功。

    Args:
        status: 求解器状态，可能为字符串或元组。

    Returns:
        True 当状态包含 'ok' 或 'optimal'，否则 False。
    """
    if isinstance(status, str):
        return status in ("ok", "optimal")
    if isinstance(status, (tuple, list)):
        return any(str(s).lower() in ("ok", "optimal") for s in status)
    return False
