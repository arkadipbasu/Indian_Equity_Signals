"""
logger_config.py
----------------
JSON-line logging setup, mirroring the @logged decorator pattern used in
the equity trading dashboard project: every decorated function call
writes one JSON record (function, args summary, duration, status) to a
rotating daily log file under logs/, plus a human-readable console line.
Safe to import from both Streamlit and plain scripts/tests.
"""

import functools
import json
import logging
import os
import time
import traceback
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

_LOGGER_NAME = "market_signal_dashboard"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on rerun)

    logger.setLevel(logging.INFO)

    file_path = os.path.join(LOG_DIR, "app.log")
    file_handler = TimedRotatingFileHandler(
        file_path, when="midnight", backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def _summarize(value, max_len: int = 180) -> str:
    try:
        s = repr(value)
    except Exception:
        s = "<unrepr-able>"
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def log_event(level: str, message: str, **fields):
    """Write one structured JSON log line directly (outside the decorator)."""
    logger = get_logger()
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level.upper(),
        "message": message,
        **fields,
    }
    line = json.dumps(record, default=str)
    if level.upper() == "ERROR":
        logger.error(line)
    elif level.upper() == "WARNING":
        logger.warning(line)
    else:
        logger.info(line)


def logged(func):
    """Decorator: logs entry/exit, duration (ms), and outcome as JSON."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start = time.time()
        arg_summary = _summarize(args[:3])
        try:
            result = func(*args, **kwargs)
            duration_ms = round((time.time() - start) * 1000, 1)
            logger.info(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": "INFO",
                "function": func.__name__,
                "status": "success",
                "duration_ms": duration_ms,
                "args": arg_summary,
            }, default=str))
            return result
        except Exception as exc:
            duration_ms = round((time.time() - start) * 1000, 1)
            logger.error(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": "ERROR",
                "function": func.__name__,
                "status": "failed",
                "duration_ms": duration_ms,
                "args": arg_summary,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            }, default=str))
            raise

    return wrapper
