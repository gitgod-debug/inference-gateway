import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from gateway.health.checker import HealthChecker
from gateway.health.circuit_breaker import CircuitBreakerManager, CircuitBreaker, State
from gateway.backends.base import BackendRegistry, BackendFactory
from gateway.models.config_models import BackendConfig
from gateway.backends.openai_compatible import OpenAICompatibleBackend

@pytest.mark.asyncio
async def test_health_checker_lifecycle_and_checks():
    registry = BackendRegistry()
    cfg1 = BackendConfig(name="b1", type="openai_compatible", base_url="http://b1")
    cfg2 = BackendConfig(name="b2", type="openai_compatible", base_url="http://b2")
    
    b1 = BackendFactory.create(cfg1)
    b2 = BackendFactory.create(cfg2)
    registry.register(b1)
    registry.register(b2)
    
    # Mock health responses
    b1.health = AsyncMock(return_value=True)
    b2.health = AsyncMock(return_value=False)
    
    checker = HealthChecker(registry, interval_seconds=0.1)
    
    # Test observers
    obs_calls = []
    def my_observer(name, healthy):
        obs_calls.append((name, healthy))
    
    # Add an observer that crashes to test error handling
    def bad_observer(name, healthy):
        raise ValueError("Observer crashed")
        
    checker.add_observer(my_observer)
    checker.add_observer(bad_observer)
    
    # Start and wait briefly
    await checker.start()
    assert checker._running is True
    assert checker._task is not None
    
    await asyncio.sleep(0.15)
    await checker.stop()
    assert checker._running is False
    
    # Both backends should have been checked
    assert b1.health.call_count >= 1
    assert b2.health.call_count >= 1
    
    # b2 was marked unhealthy
    assert registry.is_healthy("b1") is True
    assert registry.is_healthy("b2") is False
    
    # Observer should have been called for b2 (status changed from True to False)
    assert ("b2", False) in obs_calls
    # bad_observer crashed, but my_observer still ran, so observer loop didn't break.
    
    # Test timeout and exception in health check
    b1.health = AsyncMock(side_effect=TimeoutError("t"))
    b2.health = AsyncMock(side_effect=RuntimeError("r"))
    
    res = await checker.check_now()
    assert res == {"b1": False, "b2": False}

@pytest.mark.asyncio
async def test_circuit_breaker_edge_cases():
    # line 58: is_open property
    cb = CircuitBreaker(name="b1", failure_threshold=1)
    assert cb.is_open is False
    cb.record_failure()
    assert cb.is_open is True
    
    # line 89: properties in manager
    manager = CircuitBreakerManager()
    manager.can_execute("b1")
    assert manager.states["b1"] == "closed"
    manager.record_failure("b1")
    assert manager.states["b1"] == "closed" # threshold defaults to 5
