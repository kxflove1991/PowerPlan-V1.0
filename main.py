"""
main.py
-------
新能源大基地电源规划模型的主控入口。

运行方式：
    python main.py
    python main.py --config config/config.yaml --data-dir data --output-dir results
"""

import argparse
import os
import sys
from typing import Any, Dict

from src.utils import setup_logger, ConfigManager, is_solver_optimal
from src.data_processor import process_input_data, process_input_data_typical
from src.optimization_model import RenewableBaseModel
from src.validator import SystemValidator
from src.visualization import Visualizer

logger = setup_logger("Main")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="新能源大基地电源规划模型")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--data-dir", default="data", help="输入数据目录")
    parser.add_argument("--output-dir", default="results", help="输出结果目录")
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict[str, Any]:
    """加载配置文件。"""
    ConfigManager.load_config(config_path)
    return ConfigManager.get()


def prepare_data(data_dir: str, output_dir: str, n_clusters: int) -> tuple[str, str]:
    """
    准备优化和校验所需的输入数据。

    Returns:
        (optimization_input_path, validation_input_path)
    """
    wind_solar_path = os.path.join(data_dir, "Wind_Solar_Power.csv")
    load_path = os.path.join(data_dir, "load_data.csv")

    if not os.path.exists(wind_solar_path) or not os.path.exists(load_path):
        raise FileNotFoundError(f"缺少输入文件：{wind_solar_path} 或 {load_path}")

    os.makedirs(output_dir, exist_ok=True)

    opt_input = os.path.join(output_dir, "optimization_input.csv")
    val_input = os.path.join(output_dir, "validation_input.csv")

    logger.info("[Step 1] Processing input data...")
    process_input_data_typical(
        wind_solar_path, load_path,
        output_path=opt_input,
        n_clusters=n_clusters
    )
    process_input_data(
        wind_solar_path, load_path,
        output_path=val_input
    )

    return opt_input, val_input


def run_optimization(opt_input: str, output_dir: str) -> Dict[str, float]:
    """运行容量优化模型。"""
    logger.info("[Step 2] Initializing Optimization Model...")
    model = RenewableBaseModel(data_path=opt_input)
    model.build_model()

    logger.info("[Step 3] Solving...")
    status = model.solve()

    if not is_solver_optimal(status):
        raise RuntimeError(f"Optimization failed. Status: {status}")

    logger.info("[Step 4] Optimization Successful!")
    report_path = os.path.join(output_dir, "final_report.txt")
    return model.export_results(output_file=report_path)


def run_validation(val_input: str, capacities: Dict[str, float], output_dir: str) -> Dict[str, Any]:
    """使用 8760 小时数据进行独立校验。"""
    logger.info("[Step 5] Running Independent Validation (8760 hours)...")
    validator = SystemValidator(data_path=val_input)
    val_results = validator.validate(capacities)
    validator.export_detailed_results(output_dir=output_dir)
    validator.validate_constraints(val_results)
    return val_results


def run_visualization(data_dir: str, output_dir: str, n_clusters: int) -> None:
    """生成结果可视化图表。"""
    logger.info("[Step 6] Generating Visualization Figures...")
    viz = Visualizer(output_dir=os.path.join(output_dir, "figures"))

    wind_solar_path = os.path.join(data_dir, "Wind_Solar_Power.csv")
    val_dispatch = os.path.join(output_dir, "validation_hourly_dispatch.csv")
    opt_results = os.path.join(output_dir, "optimization_results.json")
    typical_dispatch = os.path.join(output_dir, "typical_day_dispatch.csv")

    viz.plot_re_8760_hourly(val_dispatch)
    if os.path.exists(wind_solar_path):
        viz.plot_typical_days_clustering(wind_solar_path, k=n_clusters)
    viz.plot_capacity_and_cost(opt_results)
    viz.plot_typical_day_dispatch(typical_dispatch)
    viz.plot_8760_system_operation(val_dispatch)

    logger.info(f"Visualization figures generated in {os.path.join(output_dir, 'figures')}/")


def append_validation_to_report(val_results: Dict[str, Any], output_dir: str) -> None:
    """将校验结果追加到最终报告。"""
    report_path = os.path.join(output_dir, "final_report.txt")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n\n====== Validation Results (8760 Hours Real Data) ======\n")
        f.write(f"Load Shedding: {val_results['total_shed_mwh']:.2f} MWh ({val_results['shed_hours']} hours)\n")
        f.write(f"Curtailment Rate: {val_results['curtailment_rate']*100:.2f}%\n")
        f.write(f"Actual Generation: {val_results['total_gen_mwh']:.2f} MWh\n")

    logger.info(f"Validation results appended to {report_path}")


def main() -> None:
    """主流程入口。"""
    logger.info(">>> Starting Renewable Base Planning Model <<<")

    args = parse_args()
    config = load_configuration(args.config)

    n_clusters = config.get("settings", {}).get("typical_days_per_month", 1)
    logger.info(f"Config loaded. Typical days per month: {n_clusters}")

    opt_input, val_input = prepare_data(args.data_dir, args.output_dir, n_clusters)
    capacities = run_optimization(opt_input, args.output_dir)
    val_results = run_validation(val_input, capacities, args.output_dir)
    append_validation_to_report(val_results, args.output_dir)
    run_visualization(args.data_dir, args.output_dir, n_clusters)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"程序运行失败: {e}")
        sys.exit(1)
