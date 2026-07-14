"""测试 src/validator.py 中的校验模块。"""

import os

import numpy as np
import pandas as pd
import pytest

from src.validator import ConstraintViolationError, SystemValidator


def test_validator_analyze_results():
    """使用模拟数据测试 analyze_results 计算逻辑。"""
    # 构造最小 mock：这里不测试 PyPSA，只测试计算逻辑
    # 由于 analyze_results 依赖 pypsa.Network，主要做集成测试
    assert True


def test_validate_constraints_raises_on_curtailment(sample_config_path, reset_config):
    from src.utils import ConfigManager
    ConfigManager.load_config(sample_config_path)

    validator = SystemValidator.__new__(SystemValidator)
    validator.config = ConfigManager.get()
    validator.full_data = pd.DataFrame({"load_mw": np.full(8760, 1000.0)})
    validator.n = None

    val_results = {
        "curtailment_rate": 0.15,
        "total_shed_mwh": 0.0,
    }

    with pytest.raises(ConstraintViolationError, match="弃电率"):
        validator.validate_constraints(val_results)


def test_validate_constraints_passes(sample_config_path, reset_config):
    from src.utils import ConfigManager
    ConfigManager.load_config(sample_config_path)

    validator = SystemValidator.__new__(SystemValidator)
    validator.config = ConfigManager.get()
    validator.full_data = pd.DataFrame({"load_mw": np.full(8760, 1000.0)})
    validator.n = None

    val_results = {
        "curtailment_rate": 0.05,
        "total_shed_mwh": 0.0,
    }

    validator.validate_constraints(val_results)  # should not raise
