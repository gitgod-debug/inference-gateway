"""Circuit breaker implementation.

State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
Prevents cascading failures by fast-failing when a backend is down.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class State(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal — requests pass through
    OPEN = "open"           # Failing — reject immediately
    HALF_OPEN = "half_open" # Testing — allow one probe request


class CircuitBreaker:
    """Per-backend circuit breaker.

    Tracks consecutive failures and opens the circuit when the
    threshold is exceeded. After a recovery timeout, allows one
    probe request (HALF_OPEN). If it succeeds, the circuit closes.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time: float = 0.0
        self.success_count = 0

    def can_execute(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == State.CLOSED:
            return True

        if self.state == State.OPEN:
            if time.monotonic() - self.last_failure_time >= self.timeout:
                self.state = State.HALF_OPEN
                logger.info("Circuit %s → HALF_OPEN (testing)", self.name)
                return True
            return False

        # HALF_OPEN — allow one probe
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = State.CLOSED
                self.failures = 0
                self.success_count = 0
                logger.info("Circuit %s → CLOSED (recovered)", self.name)
        else:
            self.failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failures += 1
        self.last_failure_time = time.monotonic()
        self.success_count = 0

        if self.state == State.HALF_OPEN:
            self.state = State.OPEN
            logger.warning("Circuit %s → OPEN (probe failed)", self.name)
        elif self.failures >= self.threshold:
            self.state = State.OPEN
            logger.warning(
                "Circuit %s → OPEN (%d failures)", self.name, self.failures
            )

    @property
    def is_open(self) -> bool:
        return self.state == State.OPEN


class CircuitBreakerManager:
    """Manages circuit breakers for all backends."""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 30.0
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._threshold = failure_threshold
        self._timeout = recovery_timeout

    def _get_breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name, failure_threshold=self._threshold,
                recovery_timeout=self._timeout,
            )
        return self._breakers[name]

    def can_execute(self, name: str) -> bool:
        return self._get_breaker(name).can_execute()

    def record_success(self, name: str) -> None:
        self._get_breaker(name).record_success()

    def record_failure(self, name: str) -> None:
        self._get_breaker(name).record_failure()

    @property
    def states(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._breakers.items()}
