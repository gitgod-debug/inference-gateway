"""Integration tests — full request lifecycle with mock backends."""

import pytest
from httpx import ASGITransport, AsyncClient

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


def _setup_app_state(app, tmp_path):
    """Manually set up app state that lifespan normally handles."""
    import gateway.backends.openai_compatible  # noqa

    routes_yaml = """
backends:
  - name: mock
    type: openai_compatible
    base_url: "http://localhost:9999"
    models: ["test-model"]
    priority: 1
routing:
  default_backend: mock
  fallback_order: [mock]
"""
    import yaml
    routes_config = RoutesConfig(**yaml.safe_load(routes_yaml))

    registry = BackendRegistry()
    for cfg in routes_config.backends:
        backend = BackendFactory.create(cfg)
        registry.register(backend)

    cb_manager = CircuitBreakerManager()
    ab_router = ABTestRouter(registry=registry, tests=[])
    canary_router = CanaryRouter(registry=registry)
    fallback_chain = FallbackChain(registry=registry,
        fallback_order=routes_config.routing.fallback_order,
        circuit_breaker_manager=cb_manager)
    request_router = RequestRouter(registry=registry, routes_config=routes_config,
        ab_test_router=ab_router, canary_router=canary_router,
        fallback_chain=fallback_chain)

    app.state.backend_registry = registry
    app.state.request_router = request_router
    app.state.fallback_chain = fallback_chain
    app.state.circuit_breaker_manager = cb_manager
    app.state.load_balancer = WeightedLoadBalancer(registry=registry)


@pytest.fixture
def integration_app(tmp_path):
    tenants = tmp_path / "tenants.yaml"
    tenants.write_text("""
tenants:
  - name: t1
    api_key: key-1
    rate_limit_rpm: 60
    allowed_backends: "*"
    tier: free
""")
    routes = tmp_path / "routes.yaml"
    routes.write_text("backends: []\nrouting:\n  default_backend: mock\n  fallback_order: [mock]")
    ab = tmp_path / "ab.yaml"
    ab.write_text("ab_tests: []")

    settings = GatewaySettings(
        auth_enabled=True, routes_config=str(routes),
        ab_tests_config=str(ab), tenants_config=str(tenants),
    )
    app = create_app(settings=settings)
    _setup_app_state(app, tmp_path)
    return app


class TestIntegration:

    @pytest.mark.asyncio
    async def test_health_endpoint(self, integration_app):
        transport = ASGITransport(app=integration_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health")
            assert r.status_code == 200
            assert "status" in r.json()

    @pytest.mark.asyncio
    async def test_models_endpoint(self, integration_app):
        transport = ASGITransport(app=integration_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/models")
            assert r.status_code == 200
            assert r.json()["object"] == "list"

    @pytest.mark.asyncio
    async def test_backends_health_endpoint(self, integration_app):
        transport = ASGITransport(app=integration_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health/backends")
            assert r.status_code == 200
            assert "backends" in r.json()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, integration_app):
        transport = ASGITransport(app=integration_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/metrics")
            assert r.status_code == 200
