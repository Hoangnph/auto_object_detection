import logging
from pathlib import Path
from typing import Optional

from ..config.config_loader import LoggingConfig

def setup_logger(config: LoggingConfig) -> None:
    """Setup application logger with file and console handlers.
    
    Args:
        config: Logging configuration
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(config.level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if config.save_to_file:
        # Create log directory if it doesn't exist
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Initial log message
    logger.info("Logger initialized") 