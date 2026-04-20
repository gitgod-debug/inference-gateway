"""Tests for circuit breaker state machine."""

import time
from unittest.mock import patch

import pytest

from gateway.health.circuit_breaker import CircuitBreaker, CircuitBreakerManager, State


class TestCircuitBreaker:
    """Test the CircuitBreaker state machine transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=5.0)
        assert cb.state == State.CLOSED
        assert cb.can_execute() is True

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.CLOSED
        assert cb.can_execute() is True

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == State.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failures == 0
        assert cb.state == State.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN

        time.sleep(0.15)
        assert cb.can_execute() is True
        assert cb.state == State.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # Transitions to HALF_OPEN

        cb.record_success()
        cb.record_success()
        assert cb.state == State.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # HALF_OPEN

        cb.record_failure()
        assert cb.state == State.OPEN


class TestCircuitBreakerManager:
    """Test the CircuitBreakerManager."""

    def test_auto_creates_breakers(self):
        mgr = CircuitBreakerManager(failure_threshold=3)
        assert mgr.can_execute("backend-1") is True
        assert "backend-1" in mgr.states

    def test_tracks_multiple_backends(self):
        mgr = CircuitBreakerManager(failure_threshold=2)
        for _ in range(2):
            mgr.record_failure("backend-a")
        mgr.record_success("backend-b")

        assert mgr.can_execute("backend-a") is False
        assert mgr.can_execute("backend-b") is True

    def test_states_property(self):
        mgr = CircuitBreakerManager(failure_threshold=1)
        mgr.record_failure("x")
        assert mgr.states["x"] == "open"
