"""
BrAIn OS — Structured Observability Logger

Provides structured JSON logging across the kernel, scheduler, agents, and API.
Uses structlog when available, with automatic fallback to standard library logging.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from brain_os.config.settings import get_settings

_initialized = False


def setup_logging() -> None:
    """Initialize structured logging based on BrAIn OS settings."""
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    log_level = getattr(logging, settings.observability.log_level.upper(), logging.INFO)

    try:
        import structlog

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if settings.observability.log_format == "json":
            formatter = structlog.processors.JSONRenderer()
        else:
            formatter = structlog.dev.ConsoleRenderer()

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processor=formatter,
            )
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

    except ImportError:
        # Fallback to standard logging with JSON-friendly format
        format_str = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            if settings.observability.log_format == "json"
            else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logging.basicConfig(level=log_level, format=format_str, stream=sys.stdout)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger instance."""
    setup_logging()
    return logging.getLogger(name)
