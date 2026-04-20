"""Canary deployment router.

Gradually shifts traffic from a stable backend to a new (canary) backend.
Automatically rolls back if the canary error rate exceeds a threshold.
"""

from __future__ import annotations

import logging
import random
import time

from gateway.backends.base import BackendRegistry, BaseHTTPBackend

logger = logging.getLogger(__name__)


class CanaryDeployment:
    """A single canary deployment definition."""

    def __init__(
        self,
        model: str,
        stable_backend: str,
        canary_backend: str,
        traffic_percent: int = 10,
        error_threshold: float = 0.1,
    ) -> None:
        self.model = model
        self.stable_backend = stable_backend
        self.canary_backend = canary_backend
        self.traffic_percent = traffic_percent
        self.error_threshold = error_threshold
        self.canary_requests = 0
        self.canary_errors = 0
        self.rolled_back = False
        self.created_at = time.monotonic()

    @property
    def error_rate(self) -> float:
        if self.canary_requests == 0:
            return 0.0
        return self.canary_errors / self.canary_requests

    def record_success(self) -> None:
        self.canary_requests += 1

    def record_failure(self) -> None:
        self.canary_requests += 1
        self.canary_errors += 1
        if self.error_rate >= self.error_threshold and self.canary_requests >= 5:
            self.rolled_back = True
            logger.warning(
                "Canary %s rolled back! Error rate %.1f%% exceeds threshold %.1f%%",
                self.canary_backend, self.error_rate * 100, self.error_threshold * 100,
            )


class CanaryRouter:
    """Routes a percentage of traffic to a canary backend.

    Monitors error rate and auto-rolls back if threshold is breached.
    """

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry
        self._deployments: dict[str, CanaryDeployment] = {}

    def add_deployment(self, deployment: CanaryDeployment) -> None:
        self._deployments[deployment.model] = deployment
        logger.info(
            "Canary deployment: %d%% traffic for model %r → %s",
            deployment.traffic_percent, deployment.model, deployment.canary_backend,
        )

    def remove_deployment(self, model: str) -> None:
        self._deployments.pop(model, None)

    def route(self, model: str) -> BaseHTTPBackend | None:
        """Route to canary or stable backend based on traffic split."""
        deployment = self._deployments.get(model)
        if deployment is None or deployment.rolled_back:
            return None

        # Percentage-based traffic split
        if random.randint(1, 100) <= deployment.traffic_percent:
            canary = self._registry.get(deployment.canary_backend)
            if canary and self._registry.is_healthy(deployment.canary_backend):
                return canary

        # Fall through to stable (handled by next router in chain)
        return None

    def record_result(self, model: str, success: bool) -> None:
        """Record canary request outcome for error tracking."""
        deployment = self._deployments.get(model)
        if deployment is None:
            return
        if success:
            deployment.record_success()
        else:
            deployment.record_failure()

    @property
    def active_deployments(self) -> dict[str, CanaryDeployment]:
        return {k: v for k, v in self._deployments.items() if not v.rolled_back}
