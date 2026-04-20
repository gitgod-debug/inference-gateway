"""Fallback chain with circuit breaker integration.

Tries backends in priority order, skipping those with open circuit breakers.
This is the last resort in the Chain of Responsibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gateway.backends.base import BackendRegistry, BaseHTTPBackend

if TYPE_CHECKING:
    from gateway.health.circuit_breaker import CircuitBreakerManager
    from gateway.models.config_models import TenantConfig

logger = logging.getLogger(__name__)


class FallbackChain:
    """Ordered fallback chain that tries backends by priority.

    Integrates with circuit breakers to skip backends that are
    experiencing failures, preventing cascading issues.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        fallback_order: list[str],
        circuit_breaker_manager: CircuitBreakerManager | None = None,
    ) -> None:
        self._registry = registry
        self._fallback_order = fallback_order
        self._cb_manager = circuit_breaker_manager

    def get_backend(
        self, tenant: TenantConfig | None = None
    ) -> BaseHTTPBackend | None:
        """Get the first healthy, accessible backend from the fallback chain."""
        for name in self._fallback_order:
            # Skip if circuit breaker is open
            if self._cb_manager and not self._cb_manager.can_execute(name):
                logger.debug("Skipping %s (circuit breaker open)", name)
                continue

            # Skip if backend is unhealthy
            if not self._registry.is_healthy(name):
                logger.debug("Skipping %s (unhealthy)", name)
                continue

            # Skip if tenant can't access this backend
            if tenant and not tenant.can_access_backend(name):
                continue

            backend = self._registry.get(name)
            if backend:
                return backend

        logger.error("All backends in fallback chain are unavailable!")
        return None

    def record_result(self, name: str, success: bool) -> None:
        """Record a request result for circuit breaker tracking."""
        if self._cb_manager:
            if success:
                self._cb_manager.record_success(name)
            else:
                self._cb_manager.record_failure(name)
