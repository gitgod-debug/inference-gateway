"""Shared test fixtures and configuration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.app import create_app
from gateway.config import GatewaySettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for all tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings(tmp_path) -> GatewaySettings:
    """Create test settings with temp config paths."""
    # Write minimal test configs
    routes = tmp_path / "routes.yaml"
    routes.write_text("""
backends:
  - name: mock-backend
    type: openai_compatible
    base_url: "http://localhost:9999"
    api_key: "test-key"
    models: ["test-model"]
    priority: 1
routing:
  default_backend: "mock-backend"
  fallback_order: ["mock-backend"]
""")

    ab_tests = tmp_path / "ab_tests.yaml"
    ab_tests.write_text("ab_tests: []")

    tenants = tmp_path / "tenants.yaml"
    tenants.write_text("""
tenants:
  - name: "test-tenant"
    api_key: "test-api-key"
    rate_limit_rpm: 120
    allowed_backends: "*"
    tier: "pro"
""")

    return GatewaySettings(
        host="127.0.0.1",
        port=8080,
        auth_enabled=True,
        log_level="debug",
        routes_config=str(routes),
        ab_tests_config=str(ab_tests),
        tenants_config=str(tenants),
    )


@pytest.fixture
async def app(settings):
    """Create the FastAPI app with test settings."""
    application = create_app(settings=settings)
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    """Async test client for the gateway."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with valid test API key."""
    return {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json",
    }
