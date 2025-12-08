import pypsa
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List
from src.utils import setup_logger, ConfigManager

logger = setup_logger("SystemValidator")

class SystemValidator:
    def __init__(self, data_path: str = 'results/validation_input.csv'):
        self.config = ConfigManager.get()
        self.data_path = data_path
        self.full_data = None
        self.n = None
        self._load_data()

    def _load_data(self):
        """Load validation data."""
        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"Validation data file not found: {self.data_path}")
        self.full_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded validation data: {self.full_data.shape}")

    def validate(self, capacities: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate the given capacities using full 8760-hour data.
        
        Args:
            capacities: Dict with keys 'Wind', 'Solar', 'Thermal', 'Storage_Power', 'Storage_Energy'
        """
        logger.info("Starting 8760-hour validation simulation...")
        
        # 1. Create Network
        n = pypsa.Network()
        n.set_snapshots(self.full_data.index)
        
        # 2. Add Components
        self._add_components(n, capacities)
        
        # 3. Solve
        logger.info("Solving dispatch problem...")
        solver = self.config['solver']['name']
        try:
             n.optimize(solver_name=solver)
        except Exception as e:
             logger.warning(f"Validation solver {solver} failed ({e}), trying automatic selection...")
             n.optimize()
             
        self.n = n
        
        # 4. Analyze
        results = self.analyze_results(n, capacities)
        return results

    def _add_components(self, n: pypsa.Network, capacities: Dict[str, float]):
        # Buses
        n.add("Bus", "Base_Bus")
        n.add("Bus", "Export_Bus")
        n.add("Bus", "Battery_Bus")
        
        # Load
        n.add("Load", "External_Load",
              bus="Export_Bus",
              p_set=self.full_data['load_mw'])
              
        # Transmission
        trans_cap = self.config['constraints']['transmission']['capacity']
        n.add("Link", "Transmission_Channel",
              bus0="Base_Bus",
              bus1="Export_Bus",
              p_nom=trans_cap,
              p_min_pu=0, p_max_pu=1, efficiency=1.0)

        # Generators
        # Wind
        n.add("Generator", "Wind",
              bus="Base_Bus",
              p_nom=capacities['Wind'],
              p_max_pu=self.full_data['wind_p_max_pu'],
              marginal_cost=0)
              
        # Solar
        n.add("Generator", "Solar",
              bus="Base_Bus",
              p_nom=capacities['Solar'],
              p_max_pu=self.full_data['solar_p_max_pu'],
              marginal_cost=0)
              
        # Thermal
        therm_min_load = self.config['constraints']['thermal']['min_load_rate']
        fuel_cost = self.config['costs']['thermal']['fuel_cost'] * 1000  # Convert to 元/MWh
        n.add("Generator", "Thermal",
              bus="Base_Bus",
              p_nom=capacities['Thermal'],
              p_min_pu=therm_min_load,
              marginal_cost=fuel_cost)
              
        # Storage
        eff = self.config['costs']['storage'].get('efficiency', 0.95)
        # capacities['Storage_Power'] is already the optimized p_nom (Battery side capacity)
        stor_p_nom = capacities['Storage_Power']
        
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

    def analyze_results(self, n: pypsa.Network, capacities: Dict[str, float]) -> Dict[str, Any]:
        # Load Shedding
        if 'Load_Shedding' in n.generators.index:
            shed_series = n.generators_t.p['Load_Shedding']
            max_shed = shed_series.max()
            total_shed = shed_series.sum()
            shed_hours = (shed_series > 0.1).sum()
            shed_events = shed_series[shed_series > 0.1]
        else:
            max_shed = 0
            total_shed = 0
            shed_hours = 0
            shed_events = pd.Series(dtype=float)

        # Curtailment
        wind_avail = n.generators_t.p_max_pu['Wind'] * capacities['Wind']
        solar_avail = n.generators_t.p_max_pu['Solar'] * capacities['Solar']
        total_avail = wind_avail.sum() + solar_avail.sum()
        
        wind_gen = n.generators_t.p['Wind']
        solar_gen = n.generators_t.p['Solar']
        total_gen = wind_gen.sum() + solar_gen.sum()
        
        curtailment_mwh = total_avail - total_gen
        curtailment_rate = curtailment_mwh / total_avail if total_avail > 0 else 0
        
        logger.info(f"[Validation] Shed Hours: {shed_hours} h, Total Shed: {total_shed:.2f} MWh")
        logger.info(f"[Validation] Curtailment Rate: {curtailment_rate*100:.2f}%")
        
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
        os.makedirs(output_dir, exist_ok=True)
            
        # Export dispatch
        p_df = self.n.generators_t.p.copy()
        
        # Add Load
        if 'External_Load' in self.n.loads_t.p_set.columns:
            p_df['Load'] = self.n.loads_t.p_set['External_Load']
            
        if 'Battery_Discharge' in self.n.links_t.p0.columns:
            p_df['Storage_Discharge'] = self.n.links_t.p0['Battery_Discharge']
        if 'Battery_Charge' in self.n.links_t.p1.columns:
            p_df['Storage_Charge'] = self.n.links_t.p1['Battery_Charge']
        if 'Battery_Store' in self.n.stores_t.e.columns:
            p_df['Storage_Level'] = self.n.stores_t.e['Battery_Store']
            
        p_df.to_csv(os.path.join(output_dir, 'validation_hourly_dispatch.csv'))
        logger.info(f"Detailed results exported to {output_dir}/validation_hourly_dispatch.csv")
