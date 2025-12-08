# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-12-08

### Refactored
- **Codebase Restructuring**: Complete refactoring of `src/` modules for better modularity.
- **Utils Module**: Added `src/utils.py` for centralized `ConfigManager` and logging setup.
- **Model Architecture**: Decomposed `RenewableBaseModel` into smaller, manageable methods.

### Fixed
- **Constraint Logic**: Fixed critical bug where policy constraints (curtailment, RE share) were skipped due to dimension name mismatches.
- **Weighting Handling**: Fixed issue with DataFrame weightings causing constraint generation failures.
- **Import Errors**: Fixed missing imports (e.g., `os`) in optimization module.

## [1.0.0] - 2025-12-07

### Added
- **Project Structure**: Established standard directory structure (`src/`, `config/`, `data/`, `results/`, `docs/`).
- **Configurable Typical Days**: Added `typical_days_per_month` setting in `config.yaml` to switch between single (fast) and multiple (accurate) typical day modes.
- **Curtailment Penalty**: Implemented configurable curtailment penalty switch (`enable_curtailment_penalty`) in `config.yaml`.
- **Unit Standardization**: Unified all cost units to PyPSA standards (converted from 元/kW to 元/MW internally) and documented in `docs/UNITS.md`.
- **Validation Module**: Integrated 8760-hour full-year simulation for rigorous result verification.
- **Git Integration**: Added `.gitignore` and prepared for version control.

### Changed
- Refactored `main.py` to use modular imports from `src` package.
- Updated `optimization_model.py` and `data_processor.py` to support new file paths.
- Moved `config.yaml` to `config/` directory.
- Moved data files to `data/` directory.

### Fixed
- Fixed unit inconsistency in cost calculations (MW vs kW scaling).
- Fixed data type mismatch errors in typical day weight calculations.
- Fixed solver configuration issues by defaulting to `highs` solver.
