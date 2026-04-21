"""Weighted load balancer with health-aware routing.

Distributes requests across multiple instances of the same backend
using weighted round-robin. Reacts to health status changes (Observer).
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.backends.base import BackendRegistry, BaseHTTPBackend

logger = logging.getLogger(__name__)


class WeightedLoadBalancer:
    """Weighted round-robin load balancer.

    Implements the Observer pattern — listens to health status changes
    from BackendRegistry and adjusts its rotation accordingly.
    """

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry
        self._counters: dict[str, itertools.cycle] = {}
        self._rebuild_needed = True

    def get_backend(self, backend_names: list[str]) -> BaseHTTPBackend | None:
        """Select the next healthy backend using round-robin."""
        healthy = [
            name for name in backend_names
            if self._registry.is_healthy(name)
        ]
        if not healthy:
            return None

        # Simple round-robin via cycling
        key = ":".join(sorted(healthy))
        if key not in self._counters:
            self._counters[key] = itertools.cycle(healthy)

        # Try up to len(healthy) times to find a healthy backend
        for _ in range(len(healthy)):
            name = next(self._counters[key])
            if self._registry.is_healthy(name):
                backend = self._registry.get(name)
                if backend:
                    return backend

        return None

    def on_health_change(self, name: str, healthy: bool) -> None:
        """Observer callback: clear cached cycles when health changes."""
        self._counters.clear()
        status = "UP" if healthy else "DOWN"
        logger.info("Load balancer notified: %s is %s, rebuilding rotation", name, status)
