"""Tests for authentication middleware."""

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

import gateway.backends.openai_compatible  # noqa
from gateway.app import create_app
from gateway.backends.base import BackendFactory, BackendRegistry
from gateway.config import GatewaySettings
from gateway.health.circuit_breaker import CircuitBreakerManager
from gateway.models.config_models import RoutesConfig
from gateway.routing.ab_test import ABTestRouter
from gateway.routing.canary import CanaryRouter
from gateway.routing.fallback import FallbackChain
from gateway.routing.load_balancer import WeightedLoadBalancer
from gateway.routing.router import RequestRouter


def _make_app(tmp_path, auth_enabled=True):
    tenants = tmp_path / "tenants.yaml"
    tenants.write_text("""
tenants:
  - name: valid-tenant
    api_key: valid-key-123
    rate_limit_rpm: 60
    allowed_backends: "*"
    tier: free
""")
    routes = tmp_path / "routes.yaml"
    routes.write_text("backends: []\nrouting:\n  default_backend: mock\n  fallback_order: [mock]")
    ab = tmp_path / "ab.yaml"
    ab.write_text("ab_tests: []")

    settings = GatewaySettings(
        auth_enabled=auth_enabled, routes_config=str(routes),
        ab_tests_config=str(ab), tenants_config=str(tenants),
    )
    app = create_app(settings=settings)

    # Manually set up app state
    rc = RoutesConfig(**yaml.safe_load(
        "backends:\n  - name: mock\n    type: openai_compatible\n    base_url: http://localhost:9999\n    models: [test]\n    priority: 1\nrouting:\n  default_backend: mock\n  fallback_order: [mock]"
    ))
    registry = BackendRegistry()
    for cfg in rc.backends:
        registry.register(BackendFactory.create(cfg))

    cb = CircuitBreakerManager()
    fc = FallbackChain(registry=registry, fallback_order=["mock"], circuit_breaker_manager=cb)
    rr = RequestRouter(registry=registry, routes_config=rc,
        ab_test_router=ABTestRouter(registry=registry, tests=[]),
        canary_router=CanaryRouter(registry=registry), fallback_chain=fc)

    app.state.backend_registry = registry
    app.state.request_router = rr
    app.state.fallback_chain = fc
    app.state.circuit_breaker_manager = cb
    app.state.load_balancer = WeightedLoadBalancer(registry=registry)
    return app


class TestAuthMiddleware:

    @pytest.mark.asyncio
    async def test_public_endpoints_bypass_auth(self, tmp_path):
        app = _make_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, tmp_path):
        app = _make_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, tmp_path):
        app = _make_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/chat/completions",
                headers={"Authorization": "Bearer wrong-key"},
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_passes_auth(self, tmp_path):
        app = _make_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/v1/chat/completions",
                headers={"Authorization": "Bearer valid-key-123"},
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]})
            # Passes auth but backend fails → 502 or 503
            assert resp.status_code in (502, 503)
