import logging
import os
from datetime import datetime

# Directory where logs will be stored
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

def get_logger() -> logging.Logger:
    """Create and return a module‑level logger.
    The logger is configured once on first call; subsequent calls return the
    same logger instance.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # Configure the root logger only once
        logging.basicConfig(
            filename=LOG_FILE_PATH,
            format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        logger.setLevel(logging.INFO)
    return logger

# Export a ready‑to‑use logger instance
logging = get_logger()