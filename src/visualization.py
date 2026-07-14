import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
import calendar
from typing import Optional, List, Dict
from src.utils import setup_logger
from src.data_processor import generate_kmeans_typical_days, _load_and_clean_ws

logger = setup_logger("Visualization")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")

class Visualizer:
    def __init__(self, output_dir: str = 'results/figures'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {path}")

    def plot_8760_source(self, df: pd.DataFrame, col_name: str, color: str, title: str, filename: str):
        """Helper for 8760 single source plot."""
        # High resolution (1920x1080 approx)
        fig, ax = plt.subplots(figsize=(16, 9)) 
        
        # Data
        ax.plot(np.arange(1, len(df)+1), df[col_name], color=color, linewidth=2, alpha=1.0)
        
        # X-axis
        ax.set_xlim(1, 8760)
        ticks = np.arange(1000, 8761, 1000)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Hour", fontsize=12)
        
        # Y-axis
        ax.set_ylabel("Power (MW)", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Grid
        ax.grid(True, which='major', color='lightgray', alpha=0.5)
        
        # Title
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.save_plot(fig, filename)

    def plot_re_8760_hourly(self, data_path: str):
        """
        Req 1: Separate Wind and Solar 8760 charts.
        """
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Normalize
        if 'Wind' not in df.columns and 'WindPower' in df.columns: df.rename(columns={'WindPower': 'Wind'}, inplace=True)
        if 'wind_p_max_pu' in df.columns: df.rename(columns={'wind_p_max_pu': 'Wind'}, inplace=True)
        
        if 'Solar' not in df.columns and 'SolarPower' in df.columns: df.rename(columns={'SolarPower': 'Solar'}, inplace=True)
        if 'solar_p_max_pu' in df.columns: df.rename(columns={'solar_p_max_pu': 'Solar'}, inplace=True)

        # 1. Wind
        if 'Wind' in df.columns:
            self.plot_8760_source(df, 'Wind', '#1f77b4', "Wind Power Output (8760 hours)", "wind_8760_hourly.png")
            
        # 2. Solar
        if 'Solar' in df.columns:
            # Use a slightly darker yellow/orange for better visibility on white background
            self.plot_8760_source(df, 'Solar', '#ffaa00', "PV Power Output (8760 hours)", "solar_8760_hourly.png")

    def plot_typical_days_clustering(self, raw_data_path: str, k: int = 4, filename_prefix: str = 'typical_day'):
        """
        Req 2 & 3: Typical Day Visualization per month.
        """
        ws_df = _load_and_clean_ws(raw_data_path)
        if ws_df is None: return
        
        typical_df = generate_kmeans_typical_days(ws_df, n_clusters=k)
        if typical_df.empty: return
        
        months = sorted(typical_df['Month'].unique())
        
        for m in months:
            month_name = calendar.month_name[m]
            month_data = typical_df[typical_df['Month'] == m]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Colors for clusters
            colors = sns.color_palette("bright", len(month_data['Day'].unique()))
            total_days_in_month = month_data['Weight'].sum()
            
            # Plot
            for idx, ((_, day_idx), group) in enumerate(month_data.groupby(['Month', 'Day'])):
                weight = group['Weight'].iloc[0]
                ratio = (weight / total_days_in_month) * 100
                
                h = group['Hour'] + 1 # 1-24
                
                # Wind
                ax.plot(h, group['WindPower'], label=f"Wind - C{day_idx} ({int(weight)}d, {ratio:.1f}%)", 
                        linestyle='-', linewidth=2.5, marker='o', markersize=4, color=colors[idx])
                # Solar
                ax.plot(h, group['SolarPower'], label=f"Solar - C{day_idx} ({int(weight)}d, {ratio:.1f}%)", 
                        linestyle='--', linewidth=2.5, marker='x', markersize=4, color=colors[idx], alpha=0.7)

            # Axis Styling
            ax.set_xlim(1, 24)
            # Show all integer ticks
            ax.set_xticks(np.arange(1, 25, 1))
            # Only label every 4 hours? Requirement says "Main tick interval 4 hours"
            # but also "Show all integer ticks". I'll label all integers but maybe small font?
            # Or assume "Show all integer ticks" means minor ticks?
            # I will set ticks at 1..24. Labels at 4, 8, 12...
            
            # Set Labels
            ax.set_xlabel("Hour")
            ax.set_ylabel("Power (MW)")
            
            # Title
            ax.set_title(f"Typical Day Output - {month_name}", fontsize=14, loc='center', fontweight='bold')
            
            # Subtitle
            subtitle = f"Clustering Method: k-means, Parameters: n_clusters={k}, per_month=True"
            ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha='center', fontsize=10, style='italic')
            
            # Legend
            ax.legend(loc='upper right', frameon=True, fontsize=10)
            
            # Text description below
            desc = (
                "Feature Analysis:\n"
                "• Seasonal Characteristics: Representative daily profile(s) for this month.\n"
                "• Complementarity: Observation of wind/solar correlation patterns.\n"
                "• Frequency: Legend indicates number of days and percentage for each cluster."
            )
            plt.subplots_adjust(bottom=0.25)
            plt.figtext(0.1, 0.05, desc, ha='left', fontsize=11, wrap=True)
            
            self.save_plot(fig, f"{filename_prefix}_{month_name}.png")

    def plot_capacity_and_cost(self, json_path: str, filename_cap: str = 'capacity_config.png', filename_cost: str = 'cost_pie.png'):
        if not os.path.exists(json_path): return
        with open(json_path, 'r') as f: data = json.load(f)
        caps = data['capacities']
        costs = data['costs']
        
        # Cap
        cap_df = pd.DataFrame([caps])
        plot_cols = [c for c in ['Wind', 'Solar', 'Thermal', 'Storage_Power'] if c in cap_df.columns]
        if not plot_cols: return
        cap_df = cap_df[plot_cols]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cap_df.plot(kind='bar', stacked=True, ax=ax, width=0.4)
        ax.set_title("Power Capacity Configuration")
        ax.set_ylabel("Capacity (MW)")
        ax.set_xticklabels(["System"], rotation=0)
        for c in ax.containers: ax.bar_label(c, label_type='center', fmt='%.0f')
        self.save_plot(fig, filename_cap)
        
        # Cost
        total_costs = {}
        if 'Wind' in caps: total_costs['Wind'] = caps['Wind'] * costs.get('wind_capex', 0)
        if 'Solar' in caps: total_costs['Solar'] = caps['Solar'] * costs.get('solar_capex', 0)
        if 'Thermal' in caps: total_costs['Thermal'] = caps['Thermal'] * costs.get('thermal_capex', 0)
        if 'Storage_Power' in caps:
            total_costs['Storage'] = (caps['Storage_Power'] * costs.get('storage_power_capex', 0) + 
                                     caps.get('Storage_Energy', 0) * costs.get('storage_energy_capex', 0))
        total_costs = {k: v for k, v in total_costs.items() if v > 0}
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(total_costs.values(), labels=total_costs.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title("Estimated CAPEX Composition")
        self.save_plot(fig, filename_cost)

    def plot_typical_day_dispatch(self, dispatch_path: str, filename: str = 'typical_day_dispatch.png'):
        if not os.path.exists(dispatch_path): return
        df = pd.read_csv(dispatch_path, index_col=0)
        if len(df) > 48: plot_df = df.iloc[:48]
        else: plot_df = df
        
        stack_cols = ['Wind', 'Solar', 'Thermal']
        if 'Storage_Discharge' in df.columns: stack_cols.append('Storage_Discharge')
        if 'Load_Shedding' in df.columns: stack_cols.append('Load_Shedding')
        stack_cols = [c for c in stack_cols if c in df.columns]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.stackplot(plot_df.index, [plot_df[c] for c in stack_cols], labels=stack_cols, alpha=0.8)
        if 'Load' in df.columns:
            ax.plot(plot_df.index, plot_df['Load'], color='black', linewidth=2, linestyle='--', label='Load')
        if 'Storage_Charge' in df.columns:
            ax.fill_between(plot_df.index, 0, -plot_df['Storage_Charge'], color='purple', alpha=0.3, label='Storage Charge')
            
        ax.set_title("Typical Day Power Dispatch (First 48h)")
        ax.set_ylabel("Power (MW)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        self.save_plot(fig, filename)

    def plot_8760_system_operation(self, dispatch_path: str, filename: str = 'system_operation_8760.png'):
        """
        Req 4: 8760 Stacked Area.
        """
        if not os.path.exists(dispatch_path): return
        df = pd.read_csv(dispatch_path, index_col=0, parse_dates=True)
        
        # Colors & Order
        colors_map = {
            'Thermal': '#555555',
            'Wind': '#1f77b4',
            'Solar': '#ff7f0e',
            'Storage_Discharge': '#2ca02c',
            'Storage_Charge': '#d62728',
            'Load': 'black'
        }
        
        # Bottom -> Top
        stack_order = ['Thermal', 'Wind', 'Solar', 'Storage_Discharge']
        
        plot_data = []
        labels = []
        colors = []
        
        for c in stack_order:
            if c in df.columns:
                plot_data.append(df[c])
                labels.append(c)
                colors.append(colors_map[c])
        
        fig, ax = plt.subplots(figsize=(16, 9))
        hours = np.arange(1, len(df) + 1)
        
        # Stacked Area (Generation)
        if plot_data:
            ax.stackplot(hours, plot_data, labels=labels, colors=colors, alpha=0.7)
            
        # Storage Charge (Negative)
        if 'Storage_Charge' in df.columns:
            ax.fill_between(hours, 0, -df['Storage_Charge'], color=colors_map['Storage_Charge'], alpha=0.7, label='Storage_Charge')
            
        # Load Curve
        if 'Load' in df.columns:
            ax.plot(hours, df['Load'], color='black', linewidth=2, label='Load', zorder=10)
            
        # Axis & Grid
        ax.set_xlim(1, 8760)
        ax.set_xticks(np.arange(1000, 8761, 1000))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (MW)")
        ax.grid(axis='y', linestyle='-', alpha=0.5)
        
        # Title
        ax.set_title("8760-Hour System Operation Overview", fontsize=14, fontweight='bold', loc='center')
        
        # Legend
        handles, _ = ax.get_legend_handles_labels()
        # Ensure correct order in legend if needed, but default is usually fine
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        self.save_plot(fig, filename)
