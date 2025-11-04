import logging
import sys
from typing import Optional

_LOGGER_NAME = "uruno_ocr_backend"


def setup_logging(level: int = logging.INFO, propagate: bool = False) -> logging.Logger:
    """Configure a root logger for the backend."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the backend namespace."""
    base = logging.getLogger(_LOGGER_NAME)
    return base if name is None else base.getChild(name)
