# import logging
# import os
# import time
# from datetime import datetime

# def setup_logger(name, log_dir, log_prefix):
#     """
#     Sets up a logger that will log to a file and console
    
#     Args:
#         name: Logger name
#         log_dir: Directory to store log files
#         log_prefix: Prefix for log file name
        
#     Returns:
#         logger: Configured logger instance
#     """
#     # Create log directory if it doesn't exist
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Create timestamp for unique log file
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
    
#     # Configure logger
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
    
#     # File handler
#     file_handler = logging.FileHandler(log_file)
#     file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(file_formatter)
    
#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(console_formatter)
    
#     # Add handlers to logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
    
#     logger.info(f"Logging to {log_file}")
    
#     return logger 


import os
import logging
import multiprocessing
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_dir, log_prefix, max_bytes=10*1024*1024, backup_count=5):
    """
    Sets up a logger that will log to a file and console with multiprocessing safety
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_prefix: Prefix for log file name
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup log files to keep
        
    Returns:
        logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
    
    # Create a lock for file access in multiprocessing
    file_lock = multiprocessing.Lock()
    
    # Custom file handler with multiprocessing safety
    class LockedRotatingFileHandler(RotatingFileHandler):
        def emit(self, record):
            with file_lock:
                super().emit(record)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to prevent duplicate logging
    logger.handlers.clear()
    
    # File handler with rotation
    file_handler = LockedRotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger