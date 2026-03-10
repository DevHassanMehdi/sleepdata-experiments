# =============================================================================
# utils/logger.py — Centralised logging setup
# =============================================================================
import logging
import sys
from pathlib import Path

# Add project root to path so config can be imported from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_logger(script_name, dataset_name):
    """Create and return a logger that writes to both console and a log file.

    Parameters
    ----------
    script_name : str
        Short name for the script, used in the log filename and logger name,
        e.g. ``"01_dataset_profiling"``.
    dataset_name : str
        Dataset being processed, e.g. ``"tihm"``.  Used in the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.  Log file is written to:
        ``logs/<script_name>_<dataset_name>.txt``

    Notes
    -----
    Calling this function multiple times with the same names is safe — existing
    handlers are cleared before new ones are added to avoid duplicate output.
    """
    # Ensure the logs directory exists
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_filename = config.LOGS_DIR / f"{script_name}_{dataset_name}.txt"
    logger_name  = f"{script_name}_{dataset_name}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent duplicate messages on re-import
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # File handler
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logger initialised → %s", log_filename)
    return logger
