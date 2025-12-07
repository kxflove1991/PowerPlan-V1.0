import os
import math
import yaml
from src.data_processor import process_input_data, process_input_data_typical
from src.optimization_model import RenewableBaseModel
from src.validator import SystemValidator

def load_config(path='config/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print(">>> 启动新能源大基地规划模型 (L1 Phase) <<<")
    
    # 0. 读取配置
    if not os.path.exists('config/config.yaml'):
        print("错误: 缺少配置文件 config/config.yaml，请先创建并填入技经数据。")
        return
    config = load_config()
    
    # 获取典型日数量设置
    n_clusters = config.get('settings', {}).get('typical_days_per_month', 1)
    print(f"配置加载: 每月典型日数量 = {n_clusters}")

    # 1. 数据检查与预处理
    print("[Step 1] 检查并处理输入数据...")
    if os.path.exists('data/Wind_Solar_Power.csv') and os.path.exists('data/load_data.csv'):
        print("使用 data/Wind_Solar_Power.csv 生成优化数据(典型日) 和 校验数据(全量)...")
        
        # 生成优化用数据 (Monthly Typical Days)
        process_input_data_typical('data/Wind_Solar_Power.csv', 'data/load_data.csv', 
                                 output_path='results/optimization_input.csv',
                                 n_clusters=n_clusters)
        
        # 生成校验用数据 (Full 8760 Real Data)
        process_input_data('data/Wind_Solar_Power.csv', 'data/load_data.csv', output_path='results/validation_input.csv')
        
        input_file = 'results/optimization_input.csv'
    else:
        print("错误: 缺少必要的数据文件 (data/Wind_Solar_Power.csv, data/load_data.csv)")
        return

    # 2. 配置检查
    # (已在开头检查)

    # 3. 初始化优化模型
    print("\n[Step 2] 初始化优化模型 (基于典型日)...")
    try:
        model = RenewableBaseModel(config_path='config/config.yaml', data_path=input_file)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 4. 构建网络
    print("[Step 3] 构建 PyPSA 能源网络...")
    try:
        model.build_model()
    except Exception as e:
        print(f"Error in build_model: {e}")
        return
    
    # 5. 求解
    print("\n[Step 4] 开始求解 (可能需要几分钟)...")
    try:
        status = model.solve()
        print(f"Solver returned status: {status}")
    except Exception as e:
        print(f"Error in solve: {e}")
        return
    
    # 6. 处理结果
    is_success = False
    if isinstance(status, str):
        if status == 'ok' or status == 'optimal':
            is_success = True
    elif isinstance(status, tuple):
        if 'ok' in status or 'optimal' in status:
            is_success = True
            
    if is_success:
        print("\n[Step 5] 优化成功! 获取最优配置方案...")
        
        # 获取优化结果
        n = model.n
        p_wind_opt = n.generators.at['Wind', 'p_nom_opt']
        p_solar_opt = n.generators.at['Solar', 'p_nom_opt']
        p_therm_opt = n.generators.at['Thermal', 'p_nom_opt'] if 'Thermal' in n.generators.index else 0
        p_stor_opt = n.links.at['Battery_Discharge', 'p_nom_opt']
        e_stor_opt = n.stores.at['Battery_Store', 'e_nom_opt']
        
        print("\n[最优电源配置]")
        print(f"  风电: {p_wind_opt:.2f} MW")
        print(f"  光伏: {p_solar_opt:.2f} MW")
        print(f"  火电: {p_therm_opt:.2f} MW")
        print(f"  储能: {p_stor_opt:.2f} MW / {e_stor_opt:.2f} MWh")
        
        # 导出优化结果
        model.export_results()
        
        # 7. 独立验证模块
        print("\n[Step 6] 启动独立验证模块 (基于 8760 小时真实数据)...")
        
        # 准备验证配置
        # 注意：如果火电是固定的，这里直接用优化结果即可 (优化结果应该等于固定值)
        # 储能能量可能因为 min_duration 约束而略有调整，验证时应保持一致
        capacities = {
            'Wind': p_wind_opt,
            'Solar': p_solar_opt,
            'Thermal': p_therm_opt,
            'Storage_Power': p_stor_opt,
            'Storage_Energy': e_stor_opt
        }
        
        try:
            validator = SystemValidator(config_path='config.yaml', data_path='validation_input.csv')
            val_results = validator.validate(capacities)
            
            # 导出详细验证结果
            validator.export_detailed_results()
            
        except Exception as e:
            print(f"[Error] 验证过程出错: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"\n[Error] 优化失败或未找到最优解. Status: {status}")

if __name__ == "__main__":
    main()
