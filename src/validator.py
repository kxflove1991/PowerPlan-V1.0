import pypsa
import pandas as pd
import numpy as np
import yaml
import os

class SystemValidator:
    def __init__(self, config_path='config.yaml', data_path='validation_input.csv'):
        self.config = self.load_config(config_path)
        self.data_path = data_path
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Validation data file not found: {data_path}")
        self.full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.n = None

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def validate(self, capacities):
        """
        使用全量 8760 小时数据验证给定装机配置的可靠性。
        capacities: dict, {'Wind': mw, 'Solar': mw, 'Thermal': mw, 'Storage_Power': mw, 'Storage_Energy': mwh}
        Returns: 
            results: dict, 包含详细的运行指标
        """
        print("\n[Validation Module] 开始进行全年 8760 小时仿真校验...")
        
        # 1. 创建验证用 Network
        n = pypsa.Network()
        n.set_snapshots(self.full_data.index)
        
        # 2. 添加组件
        n.add("Bus", "Base_Bus")
        n.add("Bus", "Export_Bus")
        
        n.add("Load", "External_Load",
              bus="Export_Bus",
              p_set=self.full_data['load_mw'])
              
        trans_cap = self.config['constraints']['transmission']['capacity']
        n.add("Link", "Transmission_Channel",
              bus0="Base_Bus",
              bus1="Export_Bus",
              p_nom=trans_cap,
              p_min_pu=0, p_max_pu=1, efficiency=1.0)

        # Generators (Fixed Capacity)
        n.add("Generator", "Wind",
              bus="Base_Bus",
              p_nom=capacities['Wind'],
              p_max_pu=self.full_data['wind_p_max_pu'],
              marginal_cost=0)
              
        n.add("Generator", "Solar",
              bus="Base_Bus",
              p_nom=capacities['Solar'],
              p_max_pu=self.full_data['solar_p_max_pu'],
              marginal_cost=0)
              
        therm_min_load = self.config['constraints']['thermal']['min_load_rate']
        n.add("Generator", "Thermal",
              bus="Base_Bus",
              p_nom=capacities['Thermal'],
              p_min_pu=therm_min_load,
              marginal_cost=100)
              
        # Storage
        n.add("Bus", "Battery_Bus")
        eff = self.config['costs']['storage'].get('efficiency', 0.95)
        
        # 电池侧容量
        stor_p_nom = capacities['Storage_Power'] / eff if eff > 0 else 0
        
        n.add("Link", "Battery_Discharge",
              bus0="Battery_Bus",
              bus1="Base_Bus",
              p_nom=stor_p_nom,
              efficiency=eff,
              marginal_cost=0)
              
        n.add("Link", "Battery_Charge",
              bus0="Base_Bus",
              bus1="Battery_Bus",
              p_nom=capacities['Storage_Power'], 
              efficiency=eff)
              
        n.add("Store", "Battery_Store",
              bus="Battery_Bus",
              e_nom=capacities['Storage_Energy'],
              e_cyclic=True)
              
        # Load Shedding
        n.add("Generator", "Load_Shedding",
              bus="Export_Bus",
              p_nom_extendable=True,
              marginal_cost=100000.0)
              
        # 3. 求解
        print("  正在执行调度模拟...")
        solver = self.config['solver']['name']
        try:
             n.optimize(solver_name=solver)
        except:
             print("  Validation solver failed, trying automatic selection...")
             n.optimize()
             
        self.n = n
        
        # 4. 结果统计
        results = self.analyze_results(n, capacities)
        return results

    def analyze_results(self, n, capacities):
        # 缺电统计
        if 'Load_Shedding' in n.generators.index:
            shed_series = n.generators_t.p['Load_Shedding']
            max_shed = shed_series.max()
            total_shed = shed_series.sum()
            shed_hours = (shed_series > 0.1).sum()
            
            # 缺电时刻
            shed_events = shed_series[shed_series > 0.1]
        else:
            max_shed = 0
            total_shed = 0
            shed_hours = 0
            shed_events = pd.Series()

        # 弃电统计
        # 理论最大出力
        wind_avail = n.generators_t.p_max_pu['Wind'] * capacities['Wind']
        solar_avail = n.generators_t.p_max_pu['Solar'] * capacities['Solar']
        total_avail = wind_avail.sum() + solar_avail.sum()
        
        # 实际出力
        wind_gen = n.generators_t.p['Wind']
        solar_gen = n.generators_t.p['Solar']
        total_gen = wind_gen.sum() + solar_gen.sum()
        
        curtailment_mwh = total_avail - total_gen
        curtailment_rate = curtailment_mwh / total_avail if total_avail > 0 else 0
        
        print(f"  [验证结果] 缺电小时: {shed_hours} h, 缺电量: {total_shed:.2f} MWh")
        print(f"  [验证结果] 弃电率: {curtailment_rate*100:.2f}%")
        
        return {
            'shed_hours': shed_hours,
            'total_shed_mwh': total_shed,
            'max_shed_mw': max_shed,
            'shed_events': shed_events,
            'curtailment_rate': curtailment_rate,
            'total_gen_mwh': total_gen,
            'total_avail_mwh': total_avail
        }
        
    def export_detailed_results(self, output_dir='results'):
        if self.n is None: return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 导出各电源出力曲线
        p_df = self.n.generators_t.p.copy()
        if 'Battery_Discharge' in self.n.links_t.p0.columns:
            p_df['Storage_Discharge'] = self.n.links_t.p0['Battery_Discharge']
        if 'Battery_Charge' in self.n.links_t.p1.columns:
            p_df['Storage_Charge'] = self.n.links_t.p1['Battery_Charge']
        if 'Battery_Store' in self.n.stores_t.e.columns:
            p_df['Storage_Level'] = self.n.stores_t.e['Battery_Store']
            
        p_df.to_csv(os.path.join(output_dir, 'validation_hourly_dispatch.csv'))
        print(f"  详细运行数据已导出至 {output_dir}/validation_hourly_dispatch.csv")
