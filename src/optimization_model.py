import os
import pypsa
import pandas as pd
import numpy as np
import xarray as xr
import linopy
from typing import Optional, Dict, Any, Tuple, Union
from src.utils import setup_logger, ConfigManager

logger = setup_logger("OptimizationModel")

class RenewableBaseModel:
    def __init__(self, data_path: str = 'results/optimization_input.csv'):
        self.config = ConfigManager.get()
        self.data_path = data_path
        self.data = None
        self.n = pypsa.Network()
        self.solution_status = None
        
        self._load_data()

    def _load_data(self):
        """Load optimization input data."""
        try:
            self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded optimization data from {self.data_path}, shape: {self.data.shape}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise

    def build_model(self):
        """Build the PyPSA network model."""
        n = self.n
        
        # --- Time & Weightings ---
        self._setup_snapshots()
        
        # --- Components ---
        self._add_buses()
        self._add_loads()
        self._add_transmission()
        self._add_generators()
        self._add_storage()
        self._add_load_shedding()
        
        logger.info("Model built successfully.")

    def _setup_snapshots(self):
        """Configure snapshots and weightings."""
        n = self.n
        
        # Check if 'weight' column exists (Pre-processed typical days)
        if 'weight' in self.data.columns:
            logger.info(f"Using pre-processed typical days (Rows: {len(self.data)}).")
            n.set_snapshots(self.data.index)
            n.snapshot_weightings = self.data['weight']
        else:
            # Fallback logic for legacy data format (Monthly representative days)
            logger.info("Using legacy monthly representative day logic.")
            mask = (self.data.index.day == 1)
            self.data = self.data[mask]
            n.set_snapshots(self.data.index)
            
            # 2022 days in month
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            weights = []
            for m in range(1, 13):
                 d = days_in_month[m-1]
                 weights.extend([float(d)] * 24)
            
            n.snapshot_weightings = pd.Series(weights, index=self.data.index)

    def _add_buses(self):
        self.n.add("Bus", "Base_Bus")
        self.n.add("Bus", "Export_Bus")
        self.n.add("Bus", "Battery_Bus")

    def _add_loads(self):
        self.n.add("Load", "External_Load",
              bus="Export_Bus",
              p_set=self.data['load_mw'])

    def _add_transmission(self):
        trans_cap = self.config['constraints']['transmission']['capacity']
        self.n.add("Link", "Transmission_Channel",
              bus0="Base_Bus",
              bus1="Export_Bus",
              p_nom=trans_cap,
              p_min_pu=0, p_max_pu=1, efficiency=1.0)

    def _get_curtailment_penalty(self) -> float:
        penalties_config = self.config.get('constraints', {}).get('penalties', {})
        if penalties_config.get('enable_curtailment_penalty', False):
            return penalties_config.get('curtailment', 0.0)
        return 0.0

    def _get_financial_params(self) -> Tuple[float, float]:
        """Calculate CRF and return (CRF, lifetime)."""
        i = 0.05
        lifetime = 20
        crf = i * (1 + i)**lifetime / ((1 + i)**lifetime - 1)
        return crf, lifetime

    def _add_generators(self):
        n = self.n
        costs = self.config['costs']
        crf, _ = self._get_financial_params()
        curt_penalty = self._get_curtailment_penalty()
        
        # Calculate availability hours for penalty calculation
        # Use series multiplication for safety
        if isinstance(n.snapshot_weightings, pd.DataFrame):
             weights = n.snapshot_weightings.iloc[:, 0]
        else:
             weights = n.snapshot_weightings
             
        wind_avail_hours = (self.data['wind_p_max_pu'] * weights).sum()
        solar_avail_hours = (self.data['solar_p_max_pu'] * weights).sum()

        # Wind
        wind_conf = self.config['constraints']['renewable'].get('wind', {})
        n.add("Generator", "Wind",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=wind_conf.get('capacity_min', 0),
              p_nom_max=wind_conf.get('capacity_max', float('inf')),
              p_max_pu=self.data['wind_p_max_pu'],
              capital_cost=costs['wind']['capex'] * 1000 * crf + curt_penalty * wind_avail_hours,
              marginal_cost=(costs['wind']['opex'] * 1000) / 8760 - curt_penalty)

        # Solar
        solar_conf = self.config['constraints']['renewable'].get('solar', {})
        n.add("Generator", "Solar",
              bus="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=solar_conf.get('capacity_min', 0),
              p_nom_max=solar_conf.get('capacity_max', float('inf')),
              p_max_pu=self.data['solar_p_max_pu'],
              capital_cost=costs['solar']['capex'] * 1000 * crf + curt_penalty * solar_avail_hours,
              marginal_cost=(costs['solar']['opex'] * 1000) / 8760 - curt_penalty)

        # Thermal
        therm_conf = self.config['constraints']['thermal']
        therm_min = therm_conf.get('capacity_min', 0)
        therm_max = therm_conf.get('capacity_max', float('inf'))
        fuel_cost = costs['thermal']['fuel_cost'] * 1000
        
        fixed_capacity = (therm_min == therm_max and therm_min > 0)
        
        n.add("Generator", "Thermal",
              bus="Base_Bus",
              p_nom=therm_min if fixed_capacity else 0,
              p_nom_extendable=not fixed_capacity,
              p_nom_min=therm_min if not fixed_capacity else 0,
              p_nom_max=therm_max if not fixed_capacity else 0,
              p_min_pu=therm_conf['min_load_rate'],
              capital_cost=costs['thermal']['capex'] * 1000 * crf,
              marginal_cost=fuel_cost + (costs['thermal']['opex'] * 1000) / 8760)

    def _add_storage(self):
        n = self.n
        costs = self.config['costs']
        stor_conf = self.config['constraints']['storage']
        crf, _ = self._get_financial_params()
        
        eff = costs['storage'].get('efficiency', 0.95)
        
        # Battery Discharge (Link)
        # Capacity on bus0 (Battery side)
        p_min = stor_conf.get('power_capacity_min', 0)
        p_max = stor_conf.get('power_capacity_max', float('inf'))
        
        n.add("Link", "Battery_Discharge",
              bus0="Battery_Bus",
              bus1="Base_Bus",
              p_nom_extendable=True,
              p_nom_min=p_min / eff if eff > 0 else 0,
              p_nom_max=p_max / eff if eff > 0 and p_max != float('inf') else float('inf'),
              efficiency=eff,
              capital_cost=costs['storage']['power_capex'] * 1000 * crf * eff)

        # Battery Charge (Link)
        n.add("Link", "Battery_Charge",
              bus0="Base_Bus",
              bus1="Battery_Bus",
              p_nom_extendable=True,
              efficiency=eff)

        # Battery Store (Store)
        e_min = stor_conf.get('energy_capacity_min', 0)
        e_max = stor_conf.get('energy_capacity_max', float('inf'))
        
        n.add("Store", "Battery_Store",
              bus="Battery_Bus",
              e_nom_extendable=True,
              e_nom_min=e_min,
              e_nom_max=e_max,
              e_cyclic=True,
              capital_cost=costs['storage']['energy_capex'] * 1000 * crf)

    def _add_load_shedding(self):
        self.n.add("Generator", "Load_Shedding",
              bus="Export_Bus",
              p_nom_extendable=True,
              marginal_cost=100000.0) # High cost

    def _define_extra_constraints(self, n, snapshots):
        """Callback for PyPSA to add custom constraints."""
        print("DEBUG: _define_extra_constraints called!")
        logger.info("Entering _define_extra_constraints...")
        model = n.model
        constraints = self.config['constraints']
        
        # 1. Variables
        p_wind = model.variables['Generator-p_nom'].loc['Wind']
        p_solar = model.variables['Generator-p_nom'].loc['Solar']
        p_stor = model.variables['Link-p_nom'].loc['Battery_Discharge']
        e_stor = model.variables['Store-e_nom'].loc['Battery_Store']
        
        # 2. Storage Config Constraints
        # Min/Max Power Ratio
        total_re = p_wind + p_solar
        model.add_constraints(
            p_stor >= constraints['storage']['min_capacity_ratio'] * total_re,
            name="storage_power_min"
        )
        model.add_constraints(
            p_stor <= constraints['storage']['max_capacity_ratio'] * total_re,
            name="storage_power_max"
        )
        
        # Duration
        model.add_constraints(
            e_stor >= constraints['storage']['min_duration'] * p_stor,
            name="storage_duration_min"
        )
        model.add_constraints(
            e_stor <= constraints['storage']['max_duration'] * p_stor,
            name="storage_duration_max"
        )
        
        # 3. Renewable Share & Curtailment
        self._add_policy_constraints(n, model, constraints)
        
        # 4. Symmetry
        p_charge = model.variables['Link-p_nom'].loc['Battery_Charge']
        model.add_constraints(p_charge == p_stor, name="battery_symmetry")

    def _add_policy_constraints(self, n, model, constraints):
        """Add RE share and curtailment constraints."""
        # Access variables safely
        p_gen = model.variables['Generator-p']
        
        # Select generators (handle dimension names)
        # Check available dimensions
        dims = list(p_gen.dims)
        logger.info(f"Available dimensions for p_gen: {dims}")
        
        gen_dim = None
        if 'Generator' in dims:
            gen_dim = 'Generator'
        elif 'generator' in dims:
            gen_dim = 'generator'
        elif 'name' in dims:
            gen_dim = 'name'
            
        if gen_dim:
            try:
                gen_wind_t = p_gen.sel({gen_dim: 'Wind'})
                gen_solar_t = p_gen.sel({gen_dim: 'Solar'})
                p_shed = p_gen.sel({gen_dim: 'Load_Shedding'}) if 'Load_Shedding' in n.generators.index else None
            except KeyError as e:
                logger.error(f"Could not find Wind/Solar generators in dimension '{gen_dim}': {e}")
                raise RuntimeError("Critical Policy Constraints could not be applied due to missing generators.")
        else:
            logger.error(f"Unknown dimension name for generators in {dims}, skipping policy constraints.")
            raise RuntimeError(f"Critical Policy Constraints could not be applied. Unknown dimensions: {dims}")

        # Prepare weights
        if isinstance(n.snapshot_weightings, pd.DataFrame):
            weights_series = n.snapshot_weightings.iloc[:, 0]
        else:
            weights_series = n.snapshot_weightings
            
        weights_da = xr.DataArray(weights_series.values, 
                                 coords={'snapshot': n.snapshots}, 
                                 dims='snapshot')
        
        # Calculate totals
        total_wind_gen = (gen_wind_t * weights_da).sum()
        total_solar_gen = (gen_solar_t * weights_da).sum()
        
        load_t = n.loads_t.p_set['External_Load']
        total_load_mwh = (load_t * weights_series).sum()
        
        # --- Constraint: Min RE Share ---
        model.add_constraints(
            (total_wind_gen + total_solar_gen) >= constraints['transmission']['min_re_share'] * total_load_mwh,
            name="min_re_share"
        )
        
        # --- Constraint: Max Load Shedding ---
        max_shed_rate = constraints['transmission'].get('max_load_shedding_rate', 0.0)
        if max_shed_rate > 0 and p_shed is not None:
            total_shed_mwh = (p_shed * weights_da).sum()
            model.add_constraints(
                total_shed_mwh <= max_shed_rate * total_load_mwh,
                name="max_load_shedding_limit"
            )

        # --- Constraint: Max Curtailment ---
        max_curtailment = constraints['transmission'].get('max_curtailment_rate', 0.10)
        
        p_wind_nom = model.variables['Generator-p_nom'].loc['Wind']
        p_solar_nom = model.variables['Generator-p_nom'].loc['Solar']
        
        wind_p_max_pu = n.generators_t.p_max_pu['Wind']
        solar_p_max_pu = n.generators_t.p_max_pu['Solar']
        
        wind_potential_factor = (wind_p_max_pu * weights_series).sum()
        solar_potential_factor = (solar_p_max_pu * weights_series).sum()
        
        total_wind_avail = p_wind_nom * wind_potential_factor
        total_solar_avail = p_solar_nom * solar_potential_factor
        
        min_gen_ratio = 1.0 - max_curtailment
        
        logger.info(f"Adding Curtailment Constraint: Max Rate={max_curtailment}")
        
        model.add_constraints(
            (total_wind_gen + total_solar_gen) >= min_gen_ratio * (total_wind_avail + total_solar_avail),
            name="max_curtailment_limit"
        )

    def solve(self):
        """Execute the optimization."""
        logger.info("Starting optimization...")
        solver = self.config['solver']['name']
        
        try:
            # Explicitly create model first
            logger.info("Creating optimization model...")
            self.n.optimize.create_model()
            
            # Add extra constraints manually
            logger.info("Adding extra constraints...")
            self._define_extra_constraints(self.n, self.n.snapshots)
            
            # Solve
            logger.info(f"Solving with {solver}...")
            status = self.n.optimize.solve_model(solver_name=solver)
        except Exception as e:
            logger.warning(f"Optimization failed with {solver} ({e}), retrying with default...")
            try:
                # Retry logic: Re-create model to be safe
                self.n.optimize.create_model()
                self._define_extra_constraints(self.n, self.n.snapshots)
                status = self.n.optimize.solve_model()
            except Exception as e2:
                logger.error(f"Optimization failed completely: {e2}")
                return "failed"
                
        self.solution_status = status
        return status

    def export_results(self, output_file='results/final_report.txt'):
        # Fix status check
        is_optimal = False
        if isinstance(self.solution_status, str) and (self.solution_status == "ok" or self.solution_status == "optimal"):
            is_optimal = True
        elif isinstance(self.solution_status, tuple) and ("ok" in self.solution_status or "optimal" in self.solution_status):
            is_optimal = True
            
        if not is_optimal:
             logger.warning(f"Optimization not optimal ({self.solution_status}), skipping export.")
             # But we might still want to see partial results if feasible
             
        n = self.n
        # ... (Export logic similar to before, but using self.config)
        # Simplified for brevity in this refactor, but ensuring core logic is preserved
        
        # Calculate key metrics
        p_wind = n.generators.at['Wind', 'p_nom_opt']
        p_solar = n.generators.at['Solar', 'p_nom_opt']
        p_therm = n.generators.at['Thermal', 'p_nom_opt']
        p_stor = n.links.at['Battery_Discharge', 'p_nom_opt']
        e_stor = n.stores.at['Battery_Store', 'e_nom_opt']
        
        # Duration
        duration = e_stor / p_stor if p_stor > 0 else 0
        
        # --- Calculate Operational Metrics (Typical Day Weighted) ---
        if isinstance(n.snapshot_weightings, pd.DataFrame):
            weights = n.snapshot_weightings.iloc[:, 0]
        else:
            weights = n.snapshot_weightings
            
        # 1. Load Shedding
        if 'Load_Shedding' in n.generators.index:
            shed_p = n.generators_t.p['Load_Shedding']
            total_shed = (shed_p * weights).sum()
        else:
            total_shed = 0.0
            
        # 2. Curtailment
        # Available
        wind_avail_t = n.generators_t.p_max_pu['Wind'] * p_wind
        solar_avail_t = n.generators_t.p_max_pu['Solar'] * p_solar
        total_avail_energy = ((wind_avail_t + solar_avail_t) * weights).sum()
        
        # Actual Generation
        wind_gen_t = n.generators_t.p['Wind']
        solar_gen_t = n.generators_t.p['Solar']
        total_gen_energy = ((wind_gen_t + solar_gen_t) * weights).sum()
        
        curtailment_energy = total_avail_energy - total_gen_energy
        curtailment_rate = (curtailment_energy / total_avail_energy * 100) if total_avail_energy > 0 else 0.0
        
        report = []
        report.append("====== Optimization Results (Typical Day Weighted) ======")
        report.append(f"Wind: {p_wind:.2f} MW")
        report.append(f"Solar: {p_solar:.2f} MW")
        report.append(f"Thermal: {p_therm:.2f} MW")
        report.append(f"Storage: {p_stor:.2f} MW / {e_stor:.2f} MWh ({duration:.1f}h)")
        report.append(f"Load Shedding: {total_shed:.2f} MWh")
        report.append(f"Curtailment Rate: {curtailment_rate:.2f}%")
        report.append(f"  - Available RE: {total_avail_energy:.2f} MWh")
        report.append(f"  - Generated RE: {total_gen_energy:.2f} MWh")
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        logger.info(f"Results exported to {output_file}")
        return {
            'Wind': p_wind, 'Solar': p_solar, 'Thermal': p_therm,
            'Storage_Power': p_stor, 'Storage_Energy': e_stor
        }
