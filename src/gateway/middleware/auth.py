"""API key authentication middleware.

Validates Bearer tokens against the tenant registry and attaches
tenant context to the request state for downstream use.

Production considerations:
  - Never leaks API key details in error messages
  - Logs failed auth attempts with client IP (security audit trail)
  - structlog import at module level (not per-request)
  - try-except around dispatch to prevent middleware crash from taking down the gateway
"""

from __future__ import annotations

import logging

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from gateway.models.config_models import TenantConfig

logger = logging.getLogger(__name__)

# Paths that don't require authentication
_PUBLIC_PATHS = {"/health", "/health/backends", "/v1/models", "/metrics", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via API key and resolve tenant.

    Security:
      - API keys are never logged or included in error responses
      - Failed auth attempts are logged with client IP for audit
    """

    def __init__(self, app, tenants: list[TenantConfig], auth_enabled: bool = True) -> None:
        super().__init__(app)
        self._auth_enabled = auth_enabled
        # Build API key → tenant lookup
        self._key_to_tenant: dict[str, TenantConfig] = {
            t.api_key: t for t in tenants
        }
        logger.info("Auth middleware initialized with %d tenants (enabled=%s)",
                     len(tenants), auth_enabled)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await self._authenticate(request, call_next)
        except Exception as e:
            # Middleware crash should NEVER take down the gateway
            logger.error(
                "Auth middleware error: %s (client=%s, path=%s)",
                e, request.client.host if request.client else "unknown",
                request.url.path, exc_info=True,
            )
            # Fail-open for internal errors (request still gets rate-limited)
            request.state.tenant = None
            return await call_next(request)

    async def _authenticate(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Core auth logic, separated for clean error handling."""
        # Skip auth for public endpoints
        if not self._auth_enabled or request.url.path in _PUBLIC_PATHS:
            request.state.tenant = None
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning("Auth failure: missing Bearer token (client=%s, path=%s)",
                          client_ip, request.url.path)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Missing or invalid Authorization header. "
                                              "Use: Authorization: Bearer <api_key>",
                                   "type": "authentication_error", "code": "invalid_api_key"}},
            )

        api_key = auth_header[7:]  # Strip "Bearer "
        tenant = self._key_to_tenant.get(api_key)

        if tenant is None:
            client_ip = request.client.host if request.client else "unknown"
            # Log ONLY that auth failed + IP, NEVER log the actual key
            logger.warning("Auth failure: invalid API key (client=%s, path=%s)",
                          client_ip, request.url.path)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key",
                                   "type": "authentication_error", "code": "invalid_api_key"}},
            )

        # Attach tenant to request state
        request.state.tenant = tenant

        structlog.contextvars.bind_contextvars(
            tenant=tenant.name, tier=tenant.tier
        )

        return await call_next(request)
