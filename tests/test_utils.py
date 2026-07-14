"""测试 src/utils.py 中的工具函数。"""

import pytest

from src.utils import ConfigManager, is_solver_optimal, setup_logger


def test_setup_logger_returns_logger():
    logger = setup_logger("TestLogger")
    assert logger.name == "TestLogger"
    assert len(logger.handlers) > 0


def test_config_manager_load(sample_config_path, reset_config):
    ConfigManager.load_config(sample_config_path)
    config = ConfigManager.get()

    assert config["settings"]["typical_days_per_month"] == 1
    assert config["financial"]["discount_rate"] == 0.05
    assert config["penalties"]["load_shedding"] == 100000.0


def test_config_manager_reset(reset_config):
    ConfigManager._config = {"test": True}
    ConfigManager.reset()
    assert ConfigManager._config is None


@pytest.mark.parametrize("status,expected", [
    ("ok", True),
    ("optimal", True),
    (("ok", "other"), True),
    (("infeasible",), False),
    ("failed", False),
    (None, False),
])
def test_is_solver_optimal(status, expected):
    assert is_solver_optimal(status) is expected
