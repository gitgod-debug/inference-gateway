"""Structured logging configuration and middleware.

Uses structlog for JSON output (production) or colored console (development).
Automatically logs request method, path, status, latency, tenant, and request_id.

Production considerations:
  - try-except in dispatch prevents logging failure from crashing requests
  - Sensitive paths (like auth headers) are never logged
  - Duration tracked even when downstream handler errors
"""

from __future__ import annotations

import logging
import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "info", is_production: bool = False) -> None:
    """Configure structlog for the application.

    Production: JSON output (for log aggregators like ELK, Datadog, CloudWatch).
    Development: Colored console output with key-value formatting.
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)


# Paths that are polled frequently (health checks, Prometheus scrape).
# Logging every scrape would flood logs with noise.
_QUIET_PATHS = {"/health", "/health/backends", "/metrics"}


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with structured metadata.

    Logs: method, path, status_code, duration_ms, client IP.
    Context vars (request_id, tenant) are automatically merged by structlog.

    Suppresses access logs for high-frequency health/metrics polling to
    prevent log explosion (~720 noise entries/hour from Prometheus alone).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        status_code = 500  # Default if we never get a response

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            _logger.error("Unhandled error in request pipeline: %s", e, exc_info=True)
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            path = request.url.path

            # Skip logging for noisy polling endpoints (health, metrics)
            # Still log if they return errors (non-2xx)
            should_log = not (path in _QUIET_PATHS and 200 <= status_code < 300)

            if should_log:
                try:
                    slog = structlog.get_logger("gateway.access")
                    await slog.ainfo(
                        "request",
                        method=request.method,
                        path=path,
                        status=status_code,
                        duration_ms=round(duration_ms, 2),
                        client=request.client.host if request.client else None,
                    )
                except Exception:
                    _logger.info(
                        "%s %s %d %.1fms",
                        request.method, path, status_code, duration_ms,
                    )
