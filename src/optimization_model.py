import pypsa
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import os
import linopy

class RenewableBaseModel:
    def __init__(self, config_path='config.yaml', data_path='clean_input_data.csv'):
        self.config = self.load_config(config_path)
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.full_data = self.data.copy() # 保存完整数据用于校验
        self.n = pypsa.Network()
        self.solution_status = None

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def build_model(self):
        n = self.n
        
        # --- 典型日聚合 (Representative Days) ---
        # 检查输入数据是否包含权重列 (意味着已经预处理为典型日格式)
        if 'weight' in self.data.columns:
            print(f"检测到预处理的典型日数据 (Rows: {len(self.data)}). 直接使用权重列.")
            n.set_snapshots(self.data.index)
            n.snapshot_weightings = self.data['weight']
        else:
            # 兼容旧模式或全量模式
            # 为了提高求解速度并解决数值稳定性问题，我们仅使用每月的一个典型日进行优化，并通过权重扩展到全年。
            # 输入数据已经是 "月典型日" 格式 (每月每天数据相同)，因此取每月1号即可代表该月。
            use_representative_days = True
            
            if use_representative_days:
                # 筛选每月 1 号的数据 (12 * 24 = 288 snapshots)
                mask = (self.data.index.day == 1)
                self.data = self.data[mask]
                n.set_snapshots(self.data.index)
                
                # 设置时间步权重 (Weightings)
                # 2022年每月天数
                days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                weights = []
                # 确保顺序对应 (假设数据按时间排序且完整覆盖1-12月)
                for m in range(1, 13):
                     # 找到该月的天数
                     d = days_in_month[m-1]
                     # 该月在 self.data 中有 24 个点
                     weights.extend([float(d)] * 24)
                
                n.snapshot_weightings = pd.Series(weights, index=self.data.index)
                print(f"已启用典型日聚合: 使用 {len(self.data)} 个时间步代表全年 (Snapshot Weightings Set).")
            else:
                n.set_snapshots(self.data.index)
        
        # 1. 节点定义
        n.add("Bus", "Base_Bus")   # 基地汇集母线
        n.add("Bus", "Export_Bus") # 外送受端母线
        
        # 2. 外送负荷 (External Load)
        # 将外送需求建模为负荷。
        n.add("Load", "External_Load",
              bus="Export_Bus",
              p_set=self.data['load_mw'])
              
        # 3. 外送通道 (Transmission Channel)
        trans_cap = self.config['constraints']['transmission']['capacity']
        n.add("Link", "Transmission_Channel",
              bus0="Base_Bus",
              bus1="Export_Bus",
              p_nom=trans_cap,
              p_min_pu=0,
              p_max_pu=1,
              efficiency=1.0) 
              
        # 4. 电源 (Generators)
        costs = self.config['costs']

        # 获取弃电惩罚参数
        penalties_config = self.config.get('constraints', {}).get('penalties', {})
        enable_penalty = penalties_config.get('enable_curtailment_penalty', False)
        curt_penalty_rate = penalties_config.get('curtailment', 0.0)
        
        curt_penalty = curt_penalty_rate if enable_penalty else 0.0
        
        # 计算理论可用发电小时数 (Availability Hours)
        # Use self.data['weight'] directly to avoid issues with n.snapshot_weightings being a DataFrame
        wind_avail_hours = (self.data['wind_p_max_pu'] * self.data['weight']).sum()
        solar_avail_hours = (self.data['solar_p_max_pu'] * self.data['weight']).sum()
        
        if curt_penalty > 0:
            print(f"已启用弃电惩罚机制: {curt_penalty} 元/MWh")
        else:
            print("弃电惩罚机制未启用。")

        # 财务参数: 资本回收系数 (CRF)
        # 假设 利率 i=5%, 寿命 n=20年
        i = 0.05
        lifetime = 20
        crf = i * (1 + i)**lifetime / ((1 + i)**lifetime - 1)
        
        # 单位修正: 
        # Config (元/kW) -> PyPSA (元/MW) => x1000
        # Capex = cost_per_kW * 1000 * crf
        # Opex = cost_per_kW * 1000 / 8760
        
        # 风电
        wind_min = self.config['constraints']['renewable'].get('wind', {}).get('capacity_min', 0)
        wind_max = self.config['constraints']['renewable'].get('wind', {}).get('capacity_max', float('inf'))
        n.add("Generator", "Wind",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=wind_min,
              p_nom_max=wind_max,
              p_max_pu=self.data['wind_p_max_pu'],
              capital_cost=costs['wind']['capex'] * 1000 * crf + curt_penalty * wind_avail_hours,
              marginal_cost=(costs['wind']['opex'] * 1000) / 8760 - curt_penalty)
              
        # 光伏
        solar_min = self.config['constraints']['renewable'].get('solar', {}).get('capacity_min', 0)
        solar_max = self.config['constraints']['renewable'].get('solar', {}).get('capacity_max', float('inf'))
        n.add("Generator", "Solar",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=solar_min,
              p_nom_max=solar_max,
              p_max_pu=self.data['solar_p_max_pu'],
              capital_cost=costs['solar']['capex'] * 1000 * crf + curt_penalty * solar_avail_hours,
              marginal_cost=(costs['solar']['opex'] * 1000) / 8760 - curt_penalty)

        # 火电 (作为灵活性调节)
        fuel_cost_mwh = costs['thermal']['fuel_cost'] * 1000
        
        # 获取火电装机约束
        therm_min = self.config['constraints']['thermal'].get('capacity_min', 0)
        therm_max = self.config['constraints']['thermal'].get('capacity_max', float('inf'))
        
        # 如果 min == max，则固定容量
        if therm_min == therm_max and therm_min > 0:
            n.add("Generator", "Thermal",
                  bus="Base_Bus",
                  p_nom=therm_min,
                  p_nom_extendable=False,
                  p_min_pu=self.config['constraints']['thermal']['min_load_rate'],
                  capital_cost=costs['thermal']['capex'] * 1000 * crf,
                  marginal_cost=fuel_cost_mwh + (costs['thermal']['opex'] * 1000) / 8760)
            print(f"  [Thermal] Fixed Capacity: {therm_min} MW")
        else:
            n.add("Generator", "Thermal",
                  bus="Base_Bus",
                  p_nom_extendable=True,
                  p_nom_min=therm_min,
                  p_nom_max=therm_max,
                  p_min_pu=self.config['constraints']['thermal']['min_load_rate'],
                  capital_cost=costs['thermal']['capex'] * 1000 * crf,
                  marginal_cost=fuel_cost_mwh + (costs['thermal']['opex'] * 1000) / 8760)

        # 5. 储能系统 (Storage)
        n.add("Bus", "Battery_Bus")
        
        # 放电链路 (代表储能功率)
        eff = costs['storage'].get('efficiency', 0.95)
        stor_p_min = self.config['constraints']['storage'].get('power_capacity_min', 0)
        stor_p_max = self.config['constraints']['storage'].get('power_capacity_max', float('inf'))
        
        # 修正: 用户定义的储能功率通常指并网点(AC侧)功率
        # PyPSA中 Link p_nom 定义在 bus0 (电池侧)，因此需要除以效率来得到电池侧所需的容量
        # 电池侧容量 = 并网侧容量 / efficiency
        # Cost adjustment: Capex (per AC kW) -> per DC MW
        # Cost = Capex_AC * P_AC = Capex_AC * (P_DC * eff) = (Capex_AC * eff) * P_DC
        n.add("Link", "Battery_Discharge",
              bus0="Battery_Bus",
              bus1="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=stor_p_min / eff if eff > 0 else 0,
              p_nom_max=stor_p_max / eff if eff > 0 and stor_p_max != float('inf') else float('inf'),
              efficiency=eff,
              capital_cost=costs['storage']['power_capex'] * 1000 * crf * eff)
              
        # 充电链路
        n.add("Link", "Battery_Charge",
              bus0="Base_Bus",
              bus1="Battery_Bus",
              p_nom_extendable=True,
              efficiency=eff)
              
        # 储能容量 (MWh)
        stor_e_min = self.config['constraints']['storage'].get('energy_capacity_min', 0)
        stor_e_max = self.config['constraints']['storage'].get('energy_capacity_max', float('inf'))
        
        n.add("Store", "Battery_Store",
              bus="Battery_Bus",
              e_nom_extendable=True,
              e_nom_min=stor_e_min,
              e_nom_max=stor_e_max,
              e_cyclic=True, # 强制日/年首尾状态一致
              capital_cost=costs['storage']['energy_capex'] * 1000 * crf)

        # 6. 缺电惩罚 (Load Shedding) - 保证模型在强制装机下可行
        n.add("Generator", "Load_Shedding",
              bus="Export_Bus",
              p_nom_extendable=True,
              marginal_cost=100000.0) # 提高缺电成本以避免非必要切负荷

    def solve(self):
        # 定义额外的耦合约束
        def extra_constraints(n, snapshots):
            model = n.model
            
            # 获取变量引用
            p_wind = model.variables['Generator-p_nom'].loc['Wind']
            p_solar = model.variables['Generator-p_nom'].loc['Solar']
            p_stor = model.variables['Link-p_nom'].loc['Battery_Discharge'] # 储能功率
            e_stor = model.variables['Store-e_nom'].loc['Battery_Store']    # 储能容量
            
            constraints = self.config['constraints']
            
            # --- 约束1: 储能功率配置比例 (15% - 20% 新能源装机) ---
            model.add_constraints(
                p_stor >= constraints['storage']['min_capacity_ratio'] * (p_wind + p_solar),
                name="storage_power_min"
            )
            model.add_constraints(
                p_stor <= constraints['storage']['max_capacity_ratio'] * (p_wind + p_solar),
                name="storage_power_max"
            )
            
            # --- 约束2: 储能时长 (2 - 4 小时) ---
            model.add_constraints(
                e_stor >= constraints['storage']['min_duration'] * p_stor,
                name="storage_duration_min"
            )
            model.add_constraints(
                e_stor <= constraints['storage']['max_duration'] * p_stor,
                name="storage_duration_max"
            )
            
            # --- 约束3: 外送电量中新能源占比 >= 50% ---
            p_gen = model.variables['Generator-p']
            print(f"[Debug] p_gen dims: {p_gen.dims}, coords: {p_gen.coords}")
            
            try:
                # 尝试适配不同的维度名称
                if 'Generator' in p_gen.dims:
                    gen_wind_t = p_gen.sel(Generator='Wind')
                    gen_solar_t = p_gen.sel(Generator='Solar')
                elif 'generator' in p_gen.dims:
                    gen_wind_t = p_gen.sel(generator='Wind')
                    gen_solar_t = p_gen.sel(generator='Solar')
                elif 'name' in p_gen.dims:
                    gen_wind_t = p_gen.sel(name='Wind')
                    gen_solar_t = p_gen.sel(name='Solar')
                else:
                    # 最后的尝试：假设第二个维度是组件名
                    print(f"[Debug] Unknown dims for p_gen, trying position-based selection")
                    # 这通常是不安全的，打印警告
                    print(f"[Warning] Cannot find Generator/generator/name dimension in p_gen")
                    
            except KeyError as e:
                print(f"[Error] Failed to select Wind/Solar generators: {e}")
                pass
            
            load_t = n.loads_t.p_set['External_Load']
            
            # Ensure variables are available before adding constraint
            if 'gen_wind_t' in locals() and 'gen_solar_t' in locals():
                 print("[Debug] Adding Renewable Share and Curtailment Constraints...")
                 # 使用权重计算总电量 (MWh)
                 # 使用 xarray DataArray 明确指定维度，避免 linopy 报错 "non-customized dimension names"
                 # Handle case where snapshot_weightings is a DataFrame (PyPSA >= 0.19)
                 if isinstance(n.snapshot_weightings, pd.DataFrame):
                     weights_series = n.snapshot_weightings['objective']
                 else:
                     weights_series = n.snapshot_weightings
                 
                 weights_da = xr.DataArray(weights_series.values, 
                                          coords={'snapshot': n.snapshots}, 
                                          dims='snapshot')
                 
                 total_wind_gen = (gen_wind_t * weights_da).sum()
                 total_solar_gen = (gen_solar_t * weights_da).sum()
                 
                 # load_t 是 pandas Series，直接与 Series 相乘即可
                 weights = n.snapshot_weightings
                 total_load_mwh = (load_t * weights).sum()
                 
                 # 临时禁用 force_dim_names 检查，因为 linopy 在处理大量项求和时可能误报 _term 维度问题
                 original_force_dim_names = model.force_dim_names
                 model.force_dim_names = False
                 
                 model.add_constraints(
                    (total_wind_gen + total_solar_gen) >= constraints['transmission']['min_re_share'] * total_load_mwh,
                    name="min_re_share"
                )
                 
                 # --- 约束4: 允许一定比例的缺电 (Max Load Shedding Rate) ---
                 max_shed_rate = constraints['transmission'].get('max_load_shedding_rate', 0.0)
                 if max_shed_rate > 0:
                     # 获取 Load Shedding 生成器的出力
                     try:
                         # 注意: 这里的 p_shed 是一个变量表达式 (Variable)
                         p_shed = p_gen.sel(Generator='Load_Shedding')
                         
                         # 使用 weights_da (DataArray) 进行加权求和
                         total_shed_mwh = (p_shed * weights_da).sum()
                         
                         model.add_constraints(
                             total_shed_mwh <= max_shed_rate * total_load_mwh,
                             name="max_load_shedding_limit"
                         )
                     except KeyError:
                         pass # 如果没有 Load_Shedding 生成器则跳过

                 # --- 约束5: 新能源弃电率 <= 10% (Max Curtailment) ---
                 max_curtailment = constraints['transmission'].get('max_curtailment_rate', 0.10)
                 
                 wind_p_max_pu = n.generators_t.p_max_pu['Wind']
                 solar_p_max_pu = n.generators_t.p_max_pu['Solar']
                 
                 # 这里都是 Pandas Series 操作，使用 weights (Series) 即可
                 total_wind_avail = p_wind * (wind_p_max_pu * weights).sum()
                 total_solar_avail = p_solar * (solar_p_max_pu * weights).sum()
                 
                 min_gen_ratio = 1.0 - max_curtailment
                 model.add_constraints(
                      (total_wind_gen + total_solar_gen) >= min_gen_ratio * (total_wind_avail + total_solar_avail),
                      name="max_curtailment_limit"
                  )
                 
                 model.force_dim_names = original_force_dim_names

            # --- 辅助约束: 充放电功率对称 ---
            p_charge = model.variables['Link-p_nom'].loc['Battery_Charge']
            model.add_constraints(p_charge == p_stor, name="battery_symmetry")

        print("开始求解优化...")
        solver = self.config['solver']['name']
        # 尝试使用指定求解器，如果失败则让 PyPSA 自动选择
        try:
            # 针对 Numpy 2.0 的兼容性检查
            import numpy as np
            if int(np.__version__.split('.')[0]) >= 2:
                print(f"检测到 Numpy {np.__version__}，如果遇到 'nonzero' 错误，请尝试降级 Numpy 或使用其他求解器。")

            status = self.n.optimize(solver_name=solver, extra_functionality=extra_constraints)
        except Exception as e:
            print(f"Warning: 指定求解器 {solver} 失败 ({e})，尝试自动选择求解器...")
            try:
                status = self.n.optimize(extra_functionality=extra_constraints)
            except Exception as e2:
                import traceback
                traceback.print_exc()
                print(f"Error: 自动求解也失败了: {e2}")
                # 如果是因为 linopy/highspy 兼容性问题，尝试给出更明确的建议
                if "nonzero" in str(e2) or "nonzero" in str(e):
                     print("\n[CRITICAL ERROR] Numpy 2.0 兼容性问题 detected.")
                     print("请尝试卸载 highspy 并安装 glpk，或者降级 numpy < 2.0")
                return "failed"
            
        self.solution_status = status
        return status

    def export_results(self):
        print("\n====== 优化结果报告 ======")
        n = self.n
        
        # 1. 装机容量
        p_wind = n.generators.at['Wind', 'p_nom_opt']
        p_solar = n.generators.at['Solar', 'p_nom_opt']
        p_therm = n.generators.at['Thermal', 'p_nom_opt']
        p_stor = n.links.at['Battery_Discharge', 'p_nom_opt']
        e_stor = n.stores.at['Battery_Store', 'e_nom_opt']
        
        # 2. 火电取整建议
        units_660 = round(p_therm / 660)
        units_1000 = round(p_therm / 1000)
        
        # 3. 关键指标校验
        if isinstance(n.snapshot_weightings, pd.DataFrame):
            weights = n.snapshot_weightings['objective']
        else:
            weights = n.snapshot_weightings
            
        total_load = (n.loads_t.p['External_Load'] * weights).sum()
        
        # 检查是否有缺电
        if 'Load_Shedding' in n.generators.index:
             total_shed = (n.generators_t.p['Load_Shedding'] * weights).sum()
        else:
             total_shed = 0
             
        real_load = total_load - total_shed
        
        re_gen = (n.generators_t.p['Wind'] * weights).sum() + (n.generators_t.p['Solar'] * weights).sum()
        chan_cap = self.config['constraints']['transmission']['capacity']
        util_hours = real_load / chan_cap
        
        # 计算新能源占比 (Renewable Share)
        # 定义: (总负荷 - 火电发电量) / 总负荷
        if 'Thermal' in n.generators.index:
             therm_gen_total = (n.generators_t.p['Thermal'] * weights).sum()
             re_share = (real_load - therm_gen_total) / real_load if real_load > 0 else 0.0
        else:
             re_share = 1.0 # 无火电则认为100% (或根据实际情况)
             
        # 计算弃电率
        # 理论发电量
        wind_potential = (n.generators_t.p_max_pu['Wind'] * weights).sum() * p_wind
        solar_potential = (n.generators_t.p_max_pu['Solar'] * weights).sum() * p_solar
        total_potential = wind_potential + solar_potential
        
        if total_potential > 0:
            curtailment_rate = 1.0 - (re_gen / total_potential)
        else:
            curtailment_rate = 0.0
        
        # 准备报告文本
        report = []
        report.append("====== 优化结果报告 ======")
        report.append("\n[推荐装机配置]")
        report.append(f"  风电: {p_wind:.2f} MW")
        report.append(f"  光伏: {p_solar:.2f} MW")
        report.append(f"  储能: {p_stor:.2f} MW / {e_stor:.2f} MWh (时长: {e_stor/p_stor if p_stor>0 else 0:.1f}h)")
        report.append(f"  火电: {p_therm:.2f} MW (理论最优)")
        report.append(f"  -> 火电工程取整建议: {units_660} 台 660MW 机组 (共 {units_660*660} MW) 或 {units_1000} 台 1000MW 机组")
        
        if total_shed > 1.0: # 忽略微小误差
            report.append(f"\n[警告] 存在缺电 (Load Shedding): {total_shed:.2f} MWh")
            report.append(f"  这表明当前装机无法满足全部外送负荷需求。")
            
            # 分析缺电时段
            shed_series = n.generators_t.p['Load_Shedding']
            shed_hours = shed_series[shed_series > 1.0]
            if not shed_hours.empty:
                max_shed = shed_hours.max()
                max_shed_time = shed_hours.idxmax()
                report.append(f"  最大缺电时刻: {max_shed_time}, 缺口: {max_shed:.2f} MW")
                report.append(f"  缺电小时数: {len(shed_hours)} 小时")
        
        report.append("\n[关键指标校验]")
        report.append(f"  新能源电量占比: {re_share*100:.2f}% (目标 >= {self.config['constraints']['transmission']['min_re_share']*100}%)")
        report.append(f"  通道利用小时数: {util_hours:.1f} 小时 (目标 >= {self.config['constraints']['transmission']['min_utilization_hours']})")
        report.append(f"  新能源弃电率:   {curtailment_rate*100:.2f}% (目标 <= {self.config['constraints']['transmission'].get('max_curtailment_rate', 0.10)*100}%)")
        
        # --- 导出详细逐时运行数据 (新增) ---
        df_result = pd.DataFrame(index=n.snapshots)
        df_result['Load'] = n.loads_t.p['External_Load']
        df_result['Wind'] = n.generators_t.p['Wind']
        df_result['Solar'] = n.generators_t.p['Solar']
        df_result['Thermal'] = n.generators_t.p['Thermal']
        df_result['Storage_Discharge'] = n.links_t.p0['Battery_Discharge']
        df_result['Storage_Charge'] = n.links_t.p1['Battery_Charge']
        df_result['Storage_Level'] = n.stores_t.e['Battery_Store']
        if 'Load_Shedding' in n.generators.index:
            df_result['Load_Shedding'] = n.generators_t.p['Load_Shedding']
        else:
            df_result['Load_Shedding'] = 0
            
        df_result.to_csv('results/dispatch_results.csv')
        print("详细运行数据已保存至 results/dispatch_results.csv")
        
        if util_hours < self.config['constraints']['transmission']['min_utilization_hours']:
             report.append("  [警告] 通道利用小时数不足！请检查负荷曲线总量或减小通道容量。")
        
        # --- 4. 经济性分析 (新增) ---
        costs = self.config['costs']
        
        # 投资成本 (Capex) - 亿元
        # 注意: p_wind 单位是 MW, capex 单位是 元/kW.  1 MW = 1000 kW.
        # 总成本 = MW * 1000 * (元/kW) / 1e8
        unit_factor = 1000.0
        
        capex_wind = p_wind * unit_factor * costs['wind']['capex'] / 1e8
        capex_solar = p_solar * unit_factor * costs['solar']['capex'] / 1e8
        capex_therm = p_therm * unit_factor * costs['thermal']['capex'] / 1e8
        capex_stor_p = p_stor * unit_factor * costs['storage']['power_capex'] / 1e8
        capex_stor_e = e_stor * unit_factor * costs['storage']['energy_capex'] / 1e8 # e_stor 是 MWh, energy_capex 是 元/kWh
        total_capex = capex_wind + capex_solar + capex_therm + capex_stor_p + capex_stor_e
        
        # 运维成本 (Opex) - 亿元/年
        opex_wind = p_wind * unit_factor * costs['wind']['opex'] / 1e8
        opex_solar = p_solar * unit_factor * costs['solar']['opex'] / 1e8
        opex_therm_fixed = p_therm * unit_factor * costs['thermal']['opex'] / 1e8
        opex_stor = p_stor * unit_factor * costs['storage']['opex'] / 1e8
        
        # 燃料成本 (Fuel Cost) - 亿元/年
        if p_therm > 0:
            therm_gen = (n.generators_t.p['Thermal'] * weights).sum() # MWh
            fuel_cost_total = therm_gen * 1000 * costs['thermal']['fuel_cost'] / 1e8 # 1000 kWh/MWh
        else:
            therm_gen = 0
            fuel_cost_total = 0
            
        total_opex_annual = opex_wind + opex_solar + opex_therm_fixed + opex_stor + fuel_cost_total
        
        # 平准化度电成本 (LCOE) - 估算
        # 假设财务参数: 资本回收系数 CRF (按 20 年, 5% 利率 ~ 0.08)
        crf = 0.0802 
        annualized_cost = total_capex * crf + total_opex_annual # 亿元/年
        
        # 总外送电量 (即负荷总量)
        total_load_mwh = total_load # MWh
        lcoe = (annualized_cost * 1e8) / (total_load_mwh * 1000) # 元/kWh (注意 load是MWh, 需转kWh; cost是元)
        
        report.append("\n[经济性分析 (估算)]")
        report.append(f"  总投资 (CAPEX): {total_capex:.2f} 亿元")
        report.append(f"    - 风电: {capex_wind:.2f} 亿元")
        report.append(f"    - 光伏: {capex_solar:.2f} 亿元")
        report.append(f"    - 火电: {capex_therm:.2f} 亿元")
        report.append(f"    - 储能: {capex_stor_p + capex_stor_e:.2f} 亿元")
        report.append(f"  年运行成本 (OPEX+燃料): {total_opex_annual:.2f} 亿元/年")
        report.append(f"  平准化度电成本 (LCOE): {lcoe:.4f} 元/kWh (含储能损耗成本)")
        report.append(f"  *注: LCOE为简易估算，假设资本回收系数(CRF)为 {crf:.4f}")

        report_text = "\n".join(report)
        print(report_text)
        
        # 保存到文件
        with open('results/final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        print("\n结果已保存至 results/final_report.txt")
