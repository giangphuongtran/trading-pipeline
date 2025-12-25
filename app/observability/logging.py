"""Structured logging setup (plain or JSON)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.config import LOG_FORMAT

_CONFIGURED = False


def _build_formatter():
    if str(LOG_FORMAT).lower() == "json":
        try:
            from pythonjsonlogger import jsonlogger
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LOG_FORMAT=json but python-json-logger is not installed. "
                "Install it (pip install python-json-logger) or set LOG_FORMAT=plain."
            ) from exc
        return jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    return logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once for CLIs/Airflow-subprocess tasks."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler()
    handler.setFormatter(_build_formatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


