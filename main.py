import os
import sys
from src.utils import setup_logger, ConfigManager
from src.data_processor import process_input_data, process_input_data_typical
from src.optimization_model import RenewableBaseModel
from src.validator import SystemValidator

logger = setup_logger("Main")

def main():
    logger.info(">>> Starting Renewable Base Planning Model <<<")
    
    # 1. Config
    config_path = 'config/config.yaml'
    try:
        ConfigManager.load_config(config_path)
        config = ConfigManager.get()
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return

    n_clusters = config.get('settings', {}).get('typical_days_per_month', 1)
    logger.info(f"Config loaded. Typical days per month: {n_clusters}")

    # 2. Data Processing
    logger.info("[Step 1] Processing input data...")
    if os.path.exists('data/Wind_Solar_Power.csv') and os.path.exists('data/load_data.csv'):
        # Optimization Data
        process_input_data_typical('data/Wind_Solar_Power.csv', 'data/load_data.csv', 
                                 output_path='results/optimization_input.csv',
                                 n_clusters=n_clusters)
        
        # Validation Data
        process_input_data('data/Wind_Solar_Power.csv', 'data/load_data.csv', 
                         output_path='results/validation_input.csv')
        
        opt_input_file = 'results/optimization_input.csv'
        val_input_file = 'results/validation_input.csv'
    else:
        logger.error("Missing input files in data/ directory.")
        return

    # 3. Optimization
    logger.info("[Step 2] Initializing Optimization Model...")
    try:
        model = RenewableBaseModel(data_path=opt_input_file)
        model.build_model()
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return
    
    logger.info("[Step 3] Solving...")
    status = model.solve()
    
    is_success = False
    if isinstance(status, str) and (status == 'ok' or status == 'optimal'):
        is_success = True
    elif isinstance(status, tuple) and ('ok' in status or 'optimal' in status):
        is_success = True
        
    if not is_success:
        logger.error(f"Optimization failed. Status: {status}")
        return
        
    logger.info("[Step 4] Optimization Successful!")
    opt_results = model.export_results()
    
    # 4. Validation
    logger.info("[Step 5] Running Independent Validation (8760 hours)...")
    try:
        validator = SystemValidator(data_path=val_input_file)
        val_results = validator.validate(opt_results)
        validator.export_detailed_results()
        
        # Update report with validation results
        _append_validation_to_report(val_results)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()

def _append_validation_to_report(val_results, report_path='results/final_report.txt'):
    """Append validation results to the final report."""
    try:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n\n====== Validation Results (8760 Hours Real Data) ======\n")
            f.write(f"Total Load: {val_results['total_avail_mwh']:.2f} MWh (Approx)\n") # Note: total_avail_mwh in val_results is Gen Avail, not Load. 
            # We should probably get load from somewhere else or just report shed/curtailment
            
            f.write(f"Load Shedding: {val_results['total_shed_mwh']:.2f} MWh ({val_results['shed_hours']} hours)\n")
            f.write(f"Curtailment Rate: {val_results['curtailment_rate']*100:.2f}%\n")
            f.write(f"Actual Generation: {val_results['total_gen_mwh']:.2f} MWh\n")
            
        logger.info(f"Validation results appended to {report_path}")
    except Exception as e:
        logger.error(f"Failed to append validation results: {e}")

if __name__ == "__main__":
    main()
