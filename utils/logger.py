"""
Structured logging configuration.
Provides a consistent logger for all modules.
"""

import logging
import sys
from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Create a structured logger with consistent formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    return logger
