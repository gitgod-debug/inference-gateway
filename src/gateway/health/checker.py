"""Background health checker for all backends.

Periodically polls backend health endpoints and updates the
BackendRegistry + Prometheus gauges.

Implements the Observer pattern: notifies registered observers
(e.g. load balancer) when health status changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from gateway.middleware.metrics import update_backend_health

if TYPE_CHECKING:
    from gateway.backends.base import BackendRegistry

logger = logging.getLogger(__name__)

# Type alias for observer callbacks
HealthObserver = Callable[[str, bool], None]


class HealthChecker:
    """Background task that polls backend health at configurable intervals.

    Observer pattern: Register callbacks via add_observer() to be notified
    when any backend's health status changes. This decouples the health checker
    from the load balancer/router — they don't need to know about each other.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        interval_seconds: float = 10.0,
    ) -> None:
        self._registry = registry
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False
        self._observers: list[HealthObserver] = []

    def add_observer(self, observer: HealthObserver) -> None:
        """Register a callback for health status changes (Observer pattern).

        The callback receives (backend_name: str, healthy: bool).
        Example:
            health_checker.add_observer(load_balancer.on_health_change)
        """
        self._observers.append(observer)

    async def start(self) -> None:
        """Start the background health check loop."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Health checker started (interval: %.1fs, observers: %d)",
                     self._interval, len(self._observers))

    async def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Health checker stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            await self._check_all()
            await asyncio.sleep(self._interval)

    async def _check_all(self) -> None:
        """Check health of all registered backends concurrently."""
        backends = self._registry.all_backends
        if not backends:
            return

        results = await asyncio.gather(
            *[self._check_one(b) for b in backends],
            return_exceptions=True,
        )

        for backend, result in zip(backends, results, strict=False):
            healthy = False if isinstance(result, Exception) else result

            # Check if status changed
            prev_healthy = self._registry.is_healthy(backend.name)
            self._registry.update_health(backend.name, healthy)
            update_backend_health(backend.name, healthy)

            # Notify observers on status CHANGE only (not every poll)
            if prev_healthy != healthy:
                self._notify_observers(backend.name, healthy)

    def _notify_observers(self, name: str, healthy: bool) -> None:
        """Notify all observers of a health status change."""
        for observer in self._observers:
            try:
                observer(name, healthy)
            except Exception as e:
                logger.error("Observer error for %s: %s", name, e)

    async def _check_one(self, backend) -> bool:
        """Check a single backend's health."""
        try:
            return await asyncio.wait_for(backend.health(), timeout=5.0)
        except TimeoutError:
            logger.warning("Health check timed out for %s", backend.name)
            return False
        except Exception as e:
            logger.warning("Health check failed for %s: %s", backend.name, e)
            return False

    async def check_now(self) -> dict[str, bool]:
        """Run an immediate health check and return results."""
        await self._check_all()
        return self._registry.health_status
