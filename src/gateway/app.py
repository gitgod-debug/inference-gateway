"""FastAPI application factory.

Builder pattern: create_app() assembles the application with all
middleware, routes, and backend initialization.

Production considerations:
  - All startup failures are caught and logged (gateway starts even with 0 backends)
  - Shutdown always runs cleanup (even on partial initialization)
  - Signal handling via lifespan context manager
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from gateway import __version__
from gateway.api.routes import router
from gateway.backends.base import BackendFactory, BackendRegistry

# Import backend modules to trigger @BackendFactory.register decorators
import gateway.backends.ollama  # noqa: F401
import gateway.backends.vllm  # noqa: F401
import gateway.backends.sglang  # noqa: F401
import gateway.backends.openai_compatible  # noqa: F401

from gateway.config import GatewaySettings, get_settings, load_yaml_config
from gateway.health.checker import HealthChecker
from gateway.health.circuit_breaker import CircuitBreakerManager
from gateway.middleware.auth import AuthMiddleware
from gateway.middleware.logging import LoggingMiddleware, configure_logging
from gateway.middleware.rate_limit import RateLimitMiddleware
from gateway.middleware.request_id import RequestIDMiddleware
from gateway.models.config_models import (
    ABTestsConfig, RoutesConfig, TenantsConfig,
)
from gateway.routing.ab_test import ABTestRouter
from gateway.routing.canary import CanaryRouter
from gateway.routing.fallback import FallbackChain
from gateway.routing.load_balancer import WeightedLoadBalancer
from gateway.routing.router import RequestRouter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize backends on startup, cleanup on shutdown.

    Production guarantees:
      - Config errors are caught and logged (gateway starts with defaults)
      - Backend init failures are isolated (one failing backend doesn't crash the gateway)
      - Shutdown always runs even on partial initialization
    """
    settings: GatewaySettings = app.state.settings

    # Track what was initialized for safe cleanup
    registry = BackendRegistry()
    health_checker = None

    try:
        # ── Load configs (graceful: bad config = empty defaults) ──
        try:
            routes_data = load_yaml_config(settings.resolve_config_path(settings.routes_config))
            routes_config = RoutesConfig(**routes_data)
        except Exception as e:
            logger.error("Failed to load routes config: %s (using empty defaults)", e)
            routes_config = RoutesConfig()

        try:
            ab_data = load_yaml_config(settings.resolve_config_path(settings.ab_tests_config))
            ab_config = ABTestsConfig(**ab_data)
        except Exception as e:
            logger.error("Failed to load A/B test config: %s (using empty defaults)", e)
            ab_config = ABTestsConfig()

        try:
            tenants_data = load_yaml_config(settings.resolve_config_path(settings.tenants_config))
            tenants_config = TenantsConfig(**tenants_data)
        except Exception as e:
            logger.error("Failed to load tenants config: %s (using empty defaults)", e)
            tenants_config = TenantsConfig()

        # ── Initialize backend registry (isolated failures per backend) ──
        for backend_cfg in routes_config.backends:
            try:
                backend = BackendFactory.create(backend_cfg)
                registry.register(backend)
            except Exception as e:
                logger.error(
                    "Failed to initialize backend %s (%s): %s — skipping",
                    backend_cfg.name, backend_cfg.type, e,
                )

        if not registry.all_backends:
            logger.warning(
                "No backends initialized! Gateway will return 503 for all requests. "
                "Check your configs/routes.yaml and .env file."
            )

        # ── Initialize routing components ───────────────────────
        cb_manager = CircuitBreakerManager(failure_threshold=5, recovery_timeout=30.0)

        ab_router = ABTestRouter(registry=registry, tests=ab_config.ab_tests)
        canary_router = CanaryRouter(registry=registry)
        fallback_chain = FallbackChain(
            registry=registry,
            fallback_order=routes_config.routing.fallback_order,
            circuit_breaker_manager=cb_manager,
        )
        load_balancer = WeightedLoadBalancer(registry=registry)

        request_router = RequestRouter(
            registry=registry,
            routes_config=routes_config,
            ab_test_router=ab_router,
            canary_router=canary_router,
            fallback_chain=fallback_chain,
        )

        # ── Store in app state ──────────────────────────────────
        app.state.backend_registry = registry
        app.state.request_router = request_router
        app.state.fallback_chain = fallback_chain
        app.state.circuit_breaker_manager = cb_manager
        app.state.load_balancer = load_balancer
        app.state.tenants_config = tenants_config

        # ── Start health checker (Observer: notifies load balancer on changes) ──
        health_checker = HealthChecker(registry=registry, interval_seconds=10.0)
        health_checker.add_observer(load_balancer.on_health_change)
        app.state.health_checker = health_checker
        await health_checker.start()

        logger.info(
            "Gateway started with %d backends: %s",
            len(registry.all_names), registry.all_names,
        )

    except Exception as e:
        # Catastrophic startup failure — log and set empty state
        logger.critical("Gateway startup failed: %s", e, exc_info=True)
        app.state.backend_registry = registry
        app.state.request_router = None
        app.state.fallback_chain = None
        app.state.circuit_breaker_manager = CircuitBreakerManager()
        app.state.load_balancer = WeightedLoadBalancer(registry)

    yield  # ── Application runs ─────────────────────────────

    # ── Shutdown (always runs, even on partial init) ─────────
    try:
        if health_checker:
            await health_checker.stop()
        await registry.close_all()
        logger.info("Gateway shut down cleanly")
    except Exception as e:
        logger.error("Error during shutdown: %s", e, exc_info=True)


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    """Application factory — builds and configures the FastAPI app.

    Builder pattern: assembles middleware, routes, and instrumentation.
    """
    if settings is None:
        settings = get_settings()

    # ── Configure logging ───────────────────────────────────
    configure_logging(
        log_level=settings.log_level,
        is_production=settings.is_production,
    )

    # ── Create app ──────────────────────────────────────────
    app = FastAPI(
        title="Inference Gateway",
        description="Unified Multi-Model Inference Gateway with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings

    # ── Load tenant config for middleware ────────────────────
    try:
        tenants_data = load_yaml_config(settings.resolve_config_path(settings.tenants_config))
        tenants_config = TenantsConfig(**tenants_data)
    except Exception as e:
        logger.error("Failed to load tenants for middleware: %s (auth disabled)", e)
        tenants_config = TenantsConfig()

    # ── Register middleware (order matters: last added = first executed) ──
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, default_rpm=settings.default_rate_limit_rpm)
    app.add_middleware(
        AuthMiddleware,
        tenants=tenants_config.tenants,
        auth_enabled=settings.auth_enabled,
    )
    app.add_middleware(RequestIDMiddleware)

    # ── Register routes ─────────────────────────────────────
    app.include_router(router)

    # ── Prometheus auto-instrumentation ─────────────────────
    try:
        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
        ).instrument(app).expose(app, endpoint="/metrics")
    except Exception as e:
        logger.error("Failed to initialize Prometheus instrumentator: %s", e)

    return app
