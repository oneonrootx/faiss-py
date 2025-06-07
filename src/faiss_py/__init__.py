import logging
import sys

# Configure logging for the package
def setup_logging(level=logging.INFO):
    """Setup logging configuration for faiss_py package."""
    logger = logging.getLogger('faiss_py')
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

# Setup default logging
logger = setup_logging()

# Make logger available at package level
__all__ = ['logger', 'setup_logging']