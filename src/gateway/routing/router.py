"""Main request router — Chain of Responsibility pattern.

Resolution order:
  1. A/B test router  (if an experiment matches the model)
  2. Canary router    (if a canary deployment is active)
  3. Model routing    (find backend that serves the requested model)
  4. Fallback chain   (ordered list of backends with circuit breakers)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gateway.backends.base import BackendRegistry, BaseHTTPBackend
from gateway.routing.ab_test import ABTestRouter
from gateway.routing.canary import CanaryRouter
from gateway.routing.fallback import FallbackChain

if TYPE_CHECKING:
    from gateway.models.config_models import RoutesConfig, TenantConfig

logger = logging.getLogger(__name__)


class RequestRouter:
    """Main router that resolves which backend handles a request.

    Implements Chain of Responsibility: each routing strategy is tried
    in order until one returns a backend. If none match, the fallback
    chain is used.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        routes_config: RoutesConfig,
        ab_test_router: ABTestRouter,
        canary_router: CanaryRouter,
        fallback_chain: FallbackChain,
    ) -> None:
        self._registry = registry
        self._routes_config = routes_config
        self._ab_test_router = ab_test_router
        self._canary_router = canary_router
        self._fallback_chain = fallback_chain
        self._default_backend = routes_config.routing.default_backend

    def resolve(
        self,
        model: str,
        tenant: TenantConfig | None = None,
        request_id: str = "",
    ) -> BaseHTTPBackend:
        """Resolve the best backend for a given model and tenant.

        Chain of Responsibility:
          1. A/B test  → weighted variant selection
          2. Canary    → gradual rollout
          3. Model map → direct model-to-backend lookup
          4. Fallback  → ordered healthy backends

        Raises ValueError if no healthy backend is found.
        """
        # ── Step 1: A/B test ────────────────────────────────
        ab_backend = self._ab_test_router.route(
            model=model,
            tenant_name=tenant.name if tenant else "",
            request_id=request_id,
        )
        if ab_backend and self._is_accessible(ab_backend, tenant):
            logger.debug("A/B test routed to %s", ab_backend.name)
            return ab_backend

        # ── Step 2: Canary ──────────────────────────────────
        canary_backend = self._canary_router.route(model=model)
        if canary_backend and self._is_accessible(canary_backend, tenant):
            logger.debug("Canary routed to %s", canary_backend.name)
            return canary_backend

        # ── Step 3: Model-based routing ─────────────────────
        model_backends = self._registry.get_backends_for_model(model)
        for backend in model_backends:
            if (self._registry.is_healthy(backend.name)
                    and self._is_accessible(backend, tenant)):
                logger.debug("Model routing → %s", backend.name)
                return backend

        # ── Step 4: Fallback chain ──────────────────────────
        fallback = self._fallback_chain.get_backend(tenant=tenant)
        if fallback:
            logger.debug("Fallback chain → %s", fallback.name)
            return fallback

        raise ValueError(
            f"No healthy backend found for model {model!r}. "
            f"All backends are down or inaccessible."
        )

    def _is_accessible(
        self, backend: BaseHTTPBackend, tenant: TenantConfig | None
    ) -> bool:
        """Check if a tenant is allowed to access a backend."""
        if tenant is None:
            return True
        return tenant.can_access_backend(backend.name)
