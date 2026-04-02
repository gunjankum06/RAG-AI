import logging
import sys

from src.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("rag_ai")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
