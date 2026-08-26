"""Structured JSON logging configuration (Phase 00.2).

Configures `structlog` to emit JSON logs, integrates with the stdlib
`logging` module (so third-party libraries and uvicorn's own loggers are
also routed through the same JSON renderer), and exposes a `request_id`
contextvar binding helper used by `app.api.middleware.RequestIDMiddleware`.

Only foundation-level logging setup lives here. Log shipping, sampling,
and any provider-specific tracing integration are out of scope for 00.2.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog + stdlib logging for JSON structured output.

    Idempotent: safe to call multiple times (e.g. once per app factory
    invocation in tests) without duplicating handlers.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Route uvicorn's own loggers through the same JSON handler instead of
    # letting them fall back to their default plain-text formatters.
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


def get_logger(name: str = "argus") -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for application use."""
    return structlog.get_logger(name)
