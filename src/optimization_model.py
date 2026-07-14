"""
optimization_model.py
---------------------
基于 PyPSA 构建新能源大基地电源容量优化模型。

主要类：
    RenewableBaseModel: 构建网络、定义约束、求解并导出结果。
"""

import json
import os
from typing import Any, Dict, Tuple, Union

import pandas as pd
import pypsa
import xarray as xr

from src.utils import setup_logger, ConfigManager, get_weight_series, is_solver_optimal

logger = setup_logger("OptimizationModel")


class RenewableBaseModel:
    """新能源大基地电源容量优化模型。"""

    def __init__(self, data_path: str = "results/optimization_input.csv"):
        self.config = ConfigManager.get()
        self.data_path = data_path
        self.data = None
        self.n = pypsa.Network()
        self.solution_status = None

        self._load_data()

    def _load_data(self):
        """加载优化输入数据。"""
        try:
            self.data = pd.read_csv(self.data_path, index_col=0)
            logger.info(f"Loaded optimization data from {self.data_path}, shape: {self.data.shape}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise

    def build_model(self):
        """构建 PyPSA 网络模型。"""
        self._setup_snapshots()
        self._add_buses()
        self._add_loads()
        self._add_transmission()
        self._add_generators()
        self._add_storage()
        self._add_load_shedding()
        logger.info("Model built successfully.")

    def _setup_snapshots(self):
        """配置 snapshots 和权重。"""
        n = self.n

        if "weight" in self.data.columns:
            logger.info(f"Using pre-processed typical days (Rows: {len(self.data)}).")
            n.set_snapshots(self.data.index)
            n.snapshot_weightings = self.data["weight"]
        else:
            logger.info("Using legacy monthly representative day logic.")
            mask = self.data.index.day == 1
            self.data = self.data[mask].copy()
            n.set_snapshots(self.data.index)

            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            weights = []
            for d in days_in_month:
                weights.extend([float(d)] * 24)

            n.snapshot_weightings = pd.Series(weights, index=self.data.index)

    def _add_buses(self):
        self.n.add("Bus", "Base_Bus")
        self.n.add("Bus", "Export_Bus")
        self.n.add("Bus", "Battery_Bus")

    def _add_loads(self):
        self.n.add("Load", "External_Load",
                   bus="Export_Bus",
                   p_set=self.data["load_mw"])

    def _add_transmission(self):
        trans_cap = self.config["constraints"]["transmission"]["capacity"]
        self.n.add("Link", "Transmission_Channel",
                   bus0="Base_Bus",
                   bus1="Export_Bus",
                   p_nom=trans_cap,
                   p_min_pu=0, p_max_pu=1, efficiency=1.0)

    def _get_curtailment_penalty(self) -> float:
        """获取弃电惩罚成本（元/MWh）。"""
        penalties_config = self.config.get("penalties", {})
        return penalties_config.get("curtailment", 0.0)

    def _get_financial_params(self) -> Tuple[float, float]:
        """计算资金回收系数 CRF 和项目寿命。"""
        fin = self.config.get("financial", {})
        i = fin.get("discount_rate", 0.05)
        lifetime = fin.get("lifetime_years", 20)
        crf = i * (1 + i) ** lifetime / ((1 + i) ** lifetime - 1)
        return crf, lifetime

    def _add_generators(self):
        n = self.n
        costs = self.config["costs"]
        crf, _ = self._get_financial_params()

        wind_conf = self.config["constraints"]["renewable"].get("wind", {})
        n.add("Generator", "Wind",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=wind_conf.get("capacity_min", 0),
              p_nom_max=wind_conf.get("capacity_max", float("inf")),
              p_max_pu=self.data["wind_p_max_pu"],
              capital_cost=costs["wind"]["capex"] * 1000 * crf,
              marginal_cost=(costs["wind"]["opex"] * 1000) / 8760)

        solar_conf = self.config["constraints"]["renewable"].get("solar", {})
        n.add("Generator", "Solar",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=solar_conf.get("capacity_min", 0),
              p_nom_max=solar_conf.get("capacity_max", float("inf")),
              p_max_pu=self.data["solar_p_max_pu"],
              capital_cost=costs["solar"]["capex"] * 1000 * crf,
              marginal_cost=(costs["solar"]["opex"] * 1000) / 8760)

        therm_conf = self.config["constraints"]["thermal"]
        therm_min = therm_conf.get("capacity_min", 0)
        therm_max = therm_conf.get("capacity_max", float("inf"))
        fuel_cost = costs["thermal"]["fuel_cost"] * 1000

        fixed_capacity = (therm_min == therm_max and therm_min > 0)

        n.add("Generator", "Thermal",
              bus="Base_Bus",
              p_nom=therm_min if fixed_capacity else 0,
              p_nom_extendable=not fixed_capacity,
              p_nom_min=therm_min if not fixed_capacity else 0,
              p_nom_max=therm_max if not fixed_capacity else 0,
              p_min_pu=therm_conf["min_load_rate"],
              capital_cost=costs["thermal"]["capex"] * 1000 * crf,
              marginal_cost=fuel_cost + (costs["thermal"]["opex"] * 1000) / 8760)

    def _add_storage(self):
        n = self.n
        costs = self.config["costs"]
        stor_conf = self.config["constraints"]["storage"]
        crf, _ = self._get_financial_params()

        eff = costs["storage"].get("efficiency", 0.95)

        p_min = stor_conf.get("power_capacity_min", 0)
        p_max = stor_conf.get("power_capacity_max", float("inf"))

        n.add("Link", "Battery_Discharge",
              bus0="Battery_Bus",
              bus1="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=p_min / eff if eff > 0 else 0,
              p_nom_max=p_max / eff if eff > 0 and p_max != float("inf") else float("inf"),
              efficiency=eff,
              capital_cost=costs["storage"]["power_capex"] * 1000 * crf * eff)

        n.add("Link", "Battery_Charge",
              bus0="Base_Bus",
              bus1="Battery_Bus",
              p_nom_extendable=True,
              efficiency=eff)

        e_min = stor_conf.get("energy_capacity_min", 0)
        e_max = stor_conf.get("energy_capacity_max", float("inf"))

        n.add("Store", "Battery_Store",
              bus="Battery_Bus",
              e_nom_extendable=True,
              e_nom_min=e_min,
              e_nom_max=e_max,
              e_cyclic=True,
              capital_cost=costs["storage"]["energy_capex"] * 1000 * crf)

    def _add_load_shedding(self):
        voll = self.config.get("penalties", {}).get("load_shedding", 100000.0)
        self.n.add("Generator", "Load_Shedding",
                   bus="Export_Bus",
                   p_nom_extendable=True,
                   marginal_cost=voll)

    def _define_extra_constraints(self, n, snapshots):
        """添加自定义约束。"""
        logger.info("Entering _define_extra_constraints...")
        model = n.model
        constraints = self.config["constraints"]

        p_wind = model.variables["Generator-p_nom"].loc["Wind"]
        p_solar = model.variables["Generator-p_nom"].loc["Solar"]
        p_stor = model.variables["Link-p_nom"].loc["Battery_Discharge"]
        e_stor = model.variables["Store-e_nom"].loc["Battery_Store"]

        # 储能功率占新能源装机比例
        total_re = p_wind + p_solar
        model.add_constraints(
            p_stor >= constraints["storage"]["min_capacity_ratio"] * total_re,
            name="storage_power_min"
        )
        model.add_constraints(
            p_stor <= constraints["storage"]["max_capacity_ratio"] * total_re,
            name="storage_power_max"
        )

        # 储能时长
        model.add_constraints(
            e_stor >= constraints["storage"]["min_duration"] * p_stor,
            name="storage_duration_min"
        )
        model.add_constraints(
            e_stor <= constraints["storage"]["max_duration"] * p_stor,
            name="storage_duration_max"
        )

        # 新能源占比与弃电率约束
        self._add_policy_constraints(n, model, constraints)

        # 充放电功率对称
        p_charge = model.variables["Link-p_nom"].loc["Battery_Charge"]
        model.add_constraints(p_charge == p_stor, name="battery_symmetry")

    def _add_policy_constraints(self, n, model, constraints):
        """添加新能源占比和弃电率约束。"""
        p_gen = model.variables["Generator-p"]

        dims = list(p_gen.dims)
        logger.info(f"Available dimensions for p_gen: {dims}")

        gen_dim = None
        for candidate in ["Generator", "generator", "name"]:
            if candidate in dims:
                gen_dim = candidate
                break

        if gen_dim is None:
            raise RuntimeError(f"Critical Policy Constraints could not be applied. Unknown dimensions: {dims}")

        try:
            gen_wind_t = p_gen.sel({gen_dim: "Wind"})
            gen_solar_t = p_gen.sel({gen_dim: "Solar"})
            p_shed = p_gen.sel({gen_dim: "Load_Shedding"}) if "Load_Shedding" in n.generators.index else None
        except KeyError as e:
            logger.error(f"Could not find Wind/Solar generators in dimension '{gen_dim}': {e}")
            raise RuntimeError("Critical Policy Constraints could not be applied due to missing generators.")

        weights_series = get_weight_series(n)
        weights_da = xr.DataArray(
            weights_series.values,
            coords={"snapshot": n.snapshots},
            dims="snapshot"
        )

        total_wind_gen = (gen_wind_t * weights_da).sum()
        total_solar_gen = (gen_solar_t * weights_da).sum()

        load_t = n.loads_t.p_set["External_Load"]
        total_load_mwh = (load_t * weights_series).sum()

        # 新能源占比下限
        model.add_constraints(
            (total_wind_gen + total_solar_gen) >= constraints["transmission"]["min_re_share"] * total_load_mwh,
            name="min_re_share"
        )

        # 最大缺电率
        max_shed_rate = constraints["transmission"].get("max_load_shedding_rate", 0.0)
        if max_shed_rate > 0 and p_shed is not None:
            total_shed_mwh = (p_shed * weights_da).sum()
            model.add_constraints(
                total_shed_mwh <= max_shed_rate * total_load_mwh,
                name="max_load_shedding_limit"
            )

        # 最大弃电率（年度累计）
        max_curtailment = constraints["transmission"].get("max_curtailment_rate", 0.10)

        p_wind_nom = model.variables["Generator-p_nom"].loc["Wind"]
        p_solar_nom = model.variables["Generator-p_nom"].loc["Solar"]

        wind_p_max_pu = n.generators_t.p_max_pu["Wind"]
        solar_p_max_pu = n.generators_t.p_max_pu["Solar"]

        # 定义显式弃电变量：curtailment(t) = available(t) - generated(t)
        curtailment = model.add_variables(
            name="curtailment",
            lower=0,
            coords=[n.snapshots]
        )

        available_re = p_wind_nom * wind_p_max_pu + p_solar_nom * solar_p_max_pu
        actual_re = gen_wind_t + gen_solar_t

        model.add_constraints(
            curtailment >= available_re - actual_re,
            name="curtailment_lower"
        )
        model.add_constraints(
            curtailment <= available_re - actual_re,
            name="curtailment_upper"
        )

        total_curtailment = (curtailment * weights_da).sum()
        total_available_re = (available_re * weights_da).sum()

        logger.info(f"Adding Curtailment Constraint: Max Rate={max_curtailment}")

        model.add_constraints(
            total_curtailment <= max_curtailment * total_available_re,
            name="max_curtailment_limit"
        )

        # 将弃电惩罚加入目标函数，进一步激励减少弃电
        penalty = self._get_curtailment_penalty()
        if penalty > 0:
            model.objective = model.objective + penalty * total_curtailment

    def solve(self) -> Union[str, Any]:
        """执行优化求解。"""
        logger.info("Starting optimization...")
        solver = self.config["solver"]["name"]

        try:
            logger.info("Creating optimization model...")
            self.n.optimize.create_model(include_objective_constant=False)

            logger.info("Adding extra constraints...")
            self._define_extra_constraints(self.n, self.n.snapshots)

            logger.info(f"Solving with {solver}...")
            status = self.n.optimize.solve_model(solver_name=solver)
        except KeyError as e:
            logger.error(f"模型变量或维度错误（可能与 PyPSA/linopy 版本有关）: {e}")
            raise
        except RuntimeError as e:
            logger.warning(f"Optimization failed with {solver} ({e}), retrying with default solver...")
            try:
                self.n.optimize.create_model(include_objective_constant=False)
                self._define_extra_constraints(self.n, self.n.snapshots)
                status = self.n.optimize.solve_model()
            except RuntimeError as e2:
                logger.error(f"Optimization failed completely: {e2}")
                raise

        self.solution_status = status
        return status

    def export_results(self, output_file: str = "results/final_report.txt") -> Dict[str, float]:
        """导出优化结果。"""
        if not is_solver_optimal(self.solution_status):
            allow = self.config.get("settings", {}).get("allow_suboptimal_export", False)
            if not allow:
                raise RuntimeError(
                    f"Optimization not optimal ({self.solution_status}), refusing to export results."
                )
            logger.warning(f"Optimization not optimal ({self.solution_status}), exporting partial results.")

        n = self.n
        p_wind = n.generators.at["Wind", "p_nom_opt"]
        p_solar = n.generators.at["Solar", "p_nom_opt"]
        p_therm = n.generators.at["Thermal", "p_nom_opt"]
        p_stor = n.links.at["Battery_Discharge", "p_nom_opt"]
        e_stor = n.stores.at["Battery_Store", "e_nom_opt"]

        duration = e_stor / p_stor if p_stor > 0 else 0

        weights = get_weight_series(n)

        # 缺电
        if "Load_Shedding" in n.generators.index:
            shed_p = n.generators_t.p["Load_Shedding"]
            total_shed = (shed_p * weights).sum()
        else:
            total_shed = 0.0

        # 弃电
        wind_avail_t = n.generators_t.p_max_pu["Wind"] * p_wind
        solar_avail_t = n.generators_t.p_max_pu["Solar"] * p_solar
        total_avail_energy = ((wind_avail_t + solar_avail_t) * weights).sum()

        wind_gen_t = n.generators_t.p["Wind"]
        solar_gen_t = n.generators_t.p["Solar"]
        therm_gen_t = n.generators_t.p["Thermal"]
        total_gen_energy = ((wind_gen_t + solar_gen_t) * weights).sum()
        total_therm_energy = (therm_gen_t * weights).sum()

        curtailment_energy = total_avail_energy - total_gen_energy
        curtailment_rate = (curtailment_energy / total_avail_energy * 100) if total_avail_energy > 0 else 0.0

        # LCOE
        costs = self.config["costs"]
        crf, _ = self._get_financial_params()

        capex_total = (
            p_wind * costs["wind"]["capex"] * 1000 * crf +
            p_solar * costs["solar"]["capex"] * 1000 * crf +
            p_therm * costs["thermal"]["capex"] * 1000 * crf +
            p_stor * costs["storage"]["power_capex"] * 1000 * crf +
            e_stor * costs["storage"]["energy_capex"] * 1000 * crf
        )

        opex_fixed_total = (
            p_wind * costs["wind"]["opex"] * 1000 +
            p_solar * costs["solar"]["opex"] * 1000 +
            p_therm * costs["thermal"]["opex"] * 1000 +
            p_stor * costs["storage"]["opex"] * 1000
        )

        fuel_cost_total = total_therm_energy * costs["thermal"]["fuel_cost"] * 1000
        total_annual_cost = capex_total + opex_fixed_total + fuel_cost_total

        load_t = n.loads_t.p_set["External_Load"]
        total_load_demand = (load_t * weights).sum()
        total_served_energy = total_load_demand - total_shed

        lcoe_rmb_per_mwh = total_annual_cost / total_served_energy if total_served_energy > 0 else float("inf")
        lcoe_rmb_per_kwh = lcoe_rmb_per_mwh / 1000

        report = [
            "====== Optimization Results (Typical Day Weighted) ======",
            f"Wind: {p_wind:.2f} MW",
            f"Solar: {p_solar:.2f} MW",
            f"Thermal: {p_therm:.2f} MW",
            f"Storage: {p_stor:.2f} MW / {e_stor:.2f} MWh ({duration:.1f}h)",
            f"Load Shedding: {total_shed:.2f} MWh",
            f"Curtailment Rate: {curtailment_rate:.2f}%",
            f"  - Available RE: {total_avail_energy:.2f} MWh",
            f"  - Generated RE: {total_gen_energy:.2f} MWh",
            "",
            "====== Economic Analysis ======",
            f"System LCOE: {lcoe_rmb_per_kwh:.4f} RMB/kWh ({lcoe_rmb_per_mwh:.2f} RMB/MWh)",
            f"Total Annual Cost: {total_annual_cost/1e8:.2f} Billion RMB",
            f"  - CAPEX (Annualized): {capex_total/1e8:.2f} Billion RMB",
            f"  - OPEX (Fixed): {opex_fixed_total/1e8:.2f} Billion RMB",
            f"  - Fuel Cost: {fuel_cost_total/1e8:.2f} Billion RMB",
        ]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        logger.info(f"Results exported to {output_file}")

        # JSON 结果
        results_dict = {
            "capacities": {
                "Wind": float(p_wind),
                "Solar": float(p_solar),
                "Thermal": float(p_therm),
                "Storage_Power": float(p_stor),
                "Storage_Energy": float(e_stor)
            },
            "metrics": {
                "load_shedding_mwh": float(total_shed),
                "curtailment_rate": float(curtailment_rate),
                "total_avail_re": float(total_avail_energy),
                "total_gen_re": float(total_gen_energy),
                "duration": float(duration),
                "lcoe_rmb_per_kwh": float(lcoe_rmb_per_kwh),
                "total_annual_cost_rmb": float(total_annual_cost)
            },
            "costs": {
                "wind_capex": costs["wind"]["capex"],
                "solar_capex": costs["solar"]["capex"],
                "thermal_capex": costs["thermal"]["capex"],
                "storage_power_capex": costs["storage"]["power_capex"],
                "storage_energy_capex": costs["storage"]["energy_capex"]
            }
        }

        json_path = os.path.join(os.path.dirname(output_file), "optimization_results.json")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        # 典型日调度 CSV
        try:
            dispatch_df = n.generators_t.p.copy()
            if "External_Load" in n.loads_t.p_set.columns:
                dispatch_df["Load"] = n.loads_t.p_set["External_Load"]

            if "Battery_Discharge" in n.links_t.p0.columns:
                dispatch_df["Storage_Discharge"] = n.links_t.p0["Battery_Discharge"]
            if "Battery_Charge" in n.links_t.p1.columns:
                dispatch_df["Storage_Charge"] = n.links_t.p1["Battery_Charge"]
            if "Battery_Store" in n.stores_t.e.columns:
                dispatch_df["Storage_Level"] = n.stores_t.e["Battery_Store"]

            dispatch_path = os.path.join(os.path.dirname(output_file), "typical_day_dispatch.csv")
            dispatch_df.to_csv(dispatch_path)
            logger.info(f"Visualization data exported to {json_path} and {dispatch_path}")
        except Exception as e:
            logger.error(f"Failed to export visualization data: {e}")

        return {
            "Wind": p_wind, "Solar": p_solar, "Thermal": p_therm,
            "Storage_Power": p_stor, "Storage_Energy": e_stor
        }
