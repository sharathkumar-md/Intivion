# Shared logging setup for all ForexGuard modules.
# Writes logs to /tmp (always writable, including Streamlit Cloud).
# Guards reconfigure() for Python versions / platforms that don't support it.

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("/tmp")


def setup_logger(name: str, log_file: str, mode: str = "a") -> logging.Logger:
    """
    Return a named logger with a stdout StreamHandler and a /tmp FileHandler.
    Safe to call multiple times — won't add duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    try:
        sh.stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, TypeError):
        pass  # not supported on this platform — fine
    logger.addHandler(sh)

    # file handler — always write to /tmp so it's writable everywhere
    try:
        fh = logging.FileHandler(_LOG_DIR / log_file, mode=mode, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # if /tmp isn't writable either, just skip file logging

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
