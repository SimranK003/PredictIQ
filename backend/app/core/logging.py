"""Centralized logging configuration.

Uses structured JSON logs so they're easy to ship to a log aggregator later.
Never log request bodies, file contents, or credentials — only metadata
(ids, counts, durations, status codes).
"""

import logging
import sys

from pythonjsonlogger import jsonlogger

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()

    if root.handlers:
        # Already configured (e.g. re-imported under uvicorn --reload)
        root.setLevel(settings.log_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Quiet noisy third-party loggers unless we're debugging.
    for noisy in ("urllib3", "botocore", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
