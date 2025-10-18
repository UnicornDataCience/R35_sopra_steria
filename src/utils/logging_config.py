import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

_configured = False

def setup_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Configure root logging once.
    Level can be overridden via LOG_LEVEL env var. Safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_name = (level or env_level).upper()
    log_level = _LEVEL_MAP.get(level_name, logging.INFO)

    logging.basicConfig(level=log_level, format=fmt or _DEFAULT_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
