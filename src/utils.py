import os
import yaml
import logging
from typing import Dict, Any, Optional

def setup_logger(name: str = "PowerPlan", level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with standard formatting.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class ConfigManager:
    """Singleton-like configuration manager."""
    _config = None

    @classmethod
    def load_config(cls, path: str = 'config/config.yaml') -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if cls._config is None:
            # Try default path if not loaded
            try:
                return cls.load_config()
            except Exception:
                raise ValueError("Configuration not loaded and default path failed. Call load_config() first.")
        return cls._config
