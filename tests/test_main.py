"""测试 main.py 的端到端流程。"""

import os
import sys

import pytest

import main
from src.utils import ConfigManager


@pytest.mark.slow
def test_main_end_to_end(temp_dir, sample_data_dir, sample_config_path, reset_config, monkeypatch):
    """运行完整主流程。"""
    output_dir = os.path.join(temp_dir, "results")

    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "--config", sample_config_path,
        "--data-dir", sample_data_dir,
        "--output-dir", output_dir,
    ])

    ConfigManager.reset()
    main.main()

    assert os.path.exists(os.path.join(output_dir, "final_report.txt"))
    assert os.path.exists(os.path.join(output_dir, "optimization_results.json"))
    assert os.path.exists(os.path.join(output_dir, "validation_hourly_dispatch.csv"))
    assert os.path.exists(os.path.join(output_dir, "figures"))
