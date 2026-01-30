import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "./logs", log_filename: str = "app.log"):
    """
    Configures logging for the entire FastAPI application.
    Rotates logs when file grows too large, keeping backups.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Define log format
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create handlers
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()

    # Common formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Root logger configuration
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
    )

    logger = logging.getLogger("sentiment_api")
    logger.info("Logging initialized successfully.")
    return logger


def get_logger(name: str = "ml_model", log_dir: str = "./logs") -> logging.Logger:
    """
    Configures and returns a logger for consistent logging across modules.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent duplicate handlers if re-imported
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger