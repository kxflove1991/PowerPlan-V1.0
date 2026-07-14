"""测试 src/optimization_model.py 中的优化模型。"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_processor import process_input_data_typical
from src.optimization_model import RenewableBaseModel
from src.utils import ConfigManager


def test_crf_calculation(sample_config_path, reset_config):
    ConfigManager.load_config(sample_config_path)
    model = RenewableBaseModel.__new__(RenewableBaseModel)
    model.config = ConfigManager.get()
    crf, lifetime = model._get_financial_params()

    expected = 0.05 * (1.05 ** 20) / ((1.05 ** 20) - 1)
    assert crf == pytest.approx(expected, rel=1e-6)
    assert lifetime == 20


def test_build_and_solve_small_model(temp_dir, sample_data_dir, sample_config_path, reset_config):
    """使用一个月的数据构建并求解一个小规模模型。"""
    ConfigManager.load_config(sample_config_path)

    ws_path = os.path.join(sample_data_dir, "Wind_Solar_Power.csv")
    load_path = os.path.join(sample_data_dir, "load_data.csv")

    opt_input = os.path.join(temp_dir, "optimization_input.csv")
    process_input_data_typical(ws_path, load_path, output_path=opt_input, n_clusters=1)

    model = RenewableBaseModel(data_path=opt_input)
    model.build_model()
    status = model.solve()

    assert status in ("ok", "optimal") or (isinstance(status, tuple) and "ok" in status)

    capacities = model.export_results(output_file=os.path.join(temp_dir, "final_report.txt"))
    assert capacities["Wind"] >= 0
    assert capacities["Solar"] >= 0
    assert capacities["Storage_Power"] >= 0
