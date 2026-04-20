"""Request ID injection middleware.

Injects a unique X-Request-ID header into every request for distributed tracing.
Binds the request ID to structlog context so all log entries include it.

Production considerations:
  - try-except ensures middleware crash never takes down the gateway
  - Validates existing X-Request-ID header (max 128 chars, prevents injection)
"""

from __future__ import annotations

import logging
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Max allowed length for incoming X-Request-ID (prevents header injection)
_MAX_REQUEST_ID_LEN = 128


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            # Use existing header or generate new UUID
            incoming_id = request.headers.get("X-Request-ID", "")
            # Validate: sanitize overly long or empty IDs
            if incoming_id and len(incoming_id) <= _MAX_REQUEST_ID_LEN:
                request_id = incoming_id
            else:
                request_id = str(uuid.uuid4())

            # Store in request state for downstream access
            request.state.request_id = request_id

            # Bind to structlog context (clear stale vars from previous request)
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)

            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            # Middleware crash should never take down the gateway
            logger.error("RequestID middleware error: %s", e, exc_info=True)
            request.state.request_id = str(uuid.uuid4())
            return await call_next(request)
