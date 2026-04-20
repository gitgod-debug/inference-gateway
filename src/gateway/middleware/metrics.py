"""Prometheus metrics middleware.

Exposes application metrics for scraping by Prometheus.
Custom metrics beyond what prometheus-fastapi-instrumentator provides.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Custom Metrics ──────────────────────────────────────────

REQUEST_COUNT = Counter(
    "gateway_request_total",
    "Total requests by backend, model, status, and tenant",
    ["backend", "model", "status", "tenant"],
)

REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Request duration in seconds by backend and model",
    ["backend", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

TOKENS_TOTAL = Counter(
    "gateway_tokens_total",
    "Total tokens processed by type and model",
    ["type", "model"],  # type: prompt | completion
)

BACKEND_HEALTH = Gauge(
    "gateway_backend_health",
    "Backend health status (1=healthy, 0=unhealthy)",
    ["backend"],
)

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Number of currently active requests per backend",
    ["backend"],
)

FALLBACK_COUNT = Counter(
    "gateway_fallback_total",
    "Number of times fallback routing was triggered",
    ["from_backend", "to_backend"],
)

AB_TEST_COUNT = Counter(
    "gateway_ab_test_total",
    "A/B test variant selections",
    ["test_name", "variant"],
)


def record_request(
    backend: str, model: str, status: int, tenant: str, duration: float
) -> None:
    """Record metrics for a completed request."""
    REQUEST_COUNT.labels(backend=backend, model=model, status=str(status), tenant=tenant).inc()
    REQUEST_DURATION.labels(backend=backend, model=model).observe(duration)


def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage metrics."""
    TOKENS_TOTAL.labels(type="prompt", model=model).inc(prompt_tokens)
    TOKENS_TOTAL.labels(type="completion", model=model).inc(completion_tokens)


def update_backend_health(backend: str, healthy: bool) -> None:
    """Update backend health gauge."""
    BACKEND_HEALTH.labels(backend=backend).set(1 if healthy else 0)
