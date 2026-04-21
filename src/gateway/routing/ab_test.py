"""A/B test traffic splitting router.

Supports weighted random selection and consistent hashing for sticky
sessions (same tenant always gets the same variant).
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.backends.base import BackendRegistry, BaseHTTPBackend
    from gateway.models.config_models import ABTestConfig

logger = logging.getLogger(__name__)


class ABTestRouter:
    """Routes requests based on A/B test configuration.

    Features:
      - Weighted random traffic splitting
      - Consistent hashing for sticky sessions (tenant → variant)
      - Per-experiment model filtering
    """

    def __init__(self, registry: BackendRegistry, tests: list[ABTestConfig]) -> None:
        self._registry = registry
        self._tests = [t for t in tests if t.enabled]
        if self._tests:
            logger.info("Active A/B tests: %s", [t.name for t in self._tests])

    def route(
        self, model: str, tenant_name: str = "", request_id: str = ""
    ) -> BaseHTTPBackend | None:
        """Try to route via an active A/B test.

        Returns None if no test matches the model.
        """
        for test in self._tests:
            if test.model != "*" and test.model != model:
                continue

            backend_name = self._select_variant(test, tenant_name, request_id)
            if backend_name:
                backend = self._registry.get(backend_name)
                if backend and self._registry.is_healthy(backend_name):
                    logger.info(
                        "A/B test %r selected variant %r",
                        test.name, backend_name,
                    )
                    return backend
        return None

    def _select_variant(
        self, test: ABTestConfig, tenant_name: str, request_id: str
    ) -> str | None:
        """Select a variant using weighted random or consistent hashing."""
        if test.sticky and tenant_name:
            return self._consistent_hash(test, tenant_name)
        return self._weighted_random(test)

    def _weighted_random(self, test: ABTestConfig) -> str | None:
        """Select a variant based on traffic weights."""
        total = sum(v.weight for v in test.variants)
        if total == 0:
            return None

        roll = random.randint(1, total)
        cumulative = 0
        for variant in test.variants:
            cumulative += variant.weight
            if roll <= cumulative:
                return variant.backend
        return test.variants[-1].backend

    def _consistent_hash(self, test: ABTestConfig, tenant_name: str) -> str | None:
        """Use consistent hashing so the same tenant always gets the same variant."""
        hash_key = f"{test.name}:{tenant_name}"
        hash_val = int(hashlib.sha256(hash_key.encode()).hexdigest(), 16)

        total = sum(v.weight for v in test.variants)
        if total == 0:
            return None

        position = hash_val % total
        cumulative = 0
        for variant in test.variants:
            cumulative += variant.weight
            if position < cumulative:
                return variant.backend
        return test.variants[-1].backend
