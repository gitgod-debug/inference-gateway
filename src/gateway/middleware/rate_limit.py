"""Token bucket rate limiter middleware.

In-memory, per-tenant rate limiting using the token bucket algorithm.
Thread-safe via asyncio.Lock.

Memory management:
  - Tenant buckets (from config) are never evicted — bounded by tenant count.
  - Anonymous IP buckets have TTL-based eviction to prevent unbounded growth.
  - Eviction runs periodically via a background check on each request.
"""

from __future__ import annotations

import asyncio
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting
_EXEMPT_PATHS = {"/health", "/health/backends", "/v1/models", "/metrics", "/docs", "/openapi.json", "/redoc"}

# Anonymous IP buckets are evicted after this many seconds of inactivity
_BUCKET_TTL_SECONDS = 600  # 10 minutes
# How often to check for stale buckets (in number of requests)
_EVICTION_INTERVAL = 100


class TokenBucket:
    """Token bucket rate limiter for a single tenant.

    Tokens refill at `rate` tokens/second up to `capacity`.
    Each request consumes 1 token.
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate            # tokens per second
        self.capacity = capacity    # max burst
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.last_used = time.monotonic()  # Track last usage for TTL eviction
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            self.last_used = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    @property
    def retry_after(self) -> float:
        """Seconds until the next token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant rate limiting middleware using token buckets.

    Memory Safety:
      - Named tenants (from tenants.yaml) → bounded by config, never evicted.
      - Anonymous IPs (no auth) → evicted after 10min inactivity to prevent
        unbounded dict growth from scanners/bots.
    """

    def __init__(self, app, default_rpm: int = 60) -> None:
        super().__init__(app)
        self._default_rpm = default_rpm
        self._buckets: dict[str, TokenBucket] = {}
        self._tenant_keys: set[str] = set()  # Tenant names — never evict these
        self._request_count = 0

    def _get_bucket(self, tenant_name: str, rpm: int, is_tenant: bool) -> TokenBucket:
        """Get or create a token bucket for a tenant/IP."""
        if tenant_name not in self._buckets:
            rate = rpm / 60.0  # Convert RPM to tokens/second
            self._buckets[tenant_name] = TokenBucket(rate=rate, capacity=rpm)
            if is_tenant:
                self._tenant_keys.add(tenant_name)
        return self._buckets[tenant_name]

    def _maybe_evict_stale_buckets(self) -> None:
        """Evict anonymous IP buckets that haven't been used recently.

        Only runs every _EVICTION_INTERVAL requests to avoid overhead.
        Never evicts named tenant buckets.
        """
        self._request_count += 1
        if self._request_count < _EVICTION_INTERVAL:
            return
        self._request_count = 0

        now = time.monotonic()
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if key not in self._tenant_keys
            and (now - bucket.last_used) > _BUCKET_TTL_SECONDS
        ]
        for key in stale_keys:
            del self._buckets[key]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Get tenant info (set by AuthMiddleware)
        tenant = getattr(request.state, "tenant", None)
        if tenant is None:
            # No tenant = use default rate limit keyed by IP
            tenant_name = request.client.host if request.client else "unknown"
            rpm = self._default_rpm
            is_tenant = False
        else:
            tenant_name = tenant.name
            rpm = tenant.rate_limit_rpm
            is_tenant = True

        bucket = self._get_bucket(tenant_name, rpm, is_tenant)

        # Periodic stale bucket eviction
        self._maybe_evict_stale_buckets()

        if not await bucket.acquire():
            retry_after = bucket.retry_after
            return JSONResponse(
                status_code=429,
                content={"error": {"message": f"Rate limit exceeded. Retry after {retry_after:.1f}s",
                                   "type": "rate_limit_error",
                                   "code": "rate_limit_exceeded"}},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)
