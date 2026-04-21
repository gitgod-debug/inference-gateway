import pytest
from unittest.mock import MagicMock

from gateway.routing.router import RequestRouter
from gateway.routing.ab_test import ABTestRouter
from gateway.routing.canary import CanaryRouter, CanaryDeployment
from gateway.routing.fallback import FallbackChain
from gateway.routing.load_balancer import WeightedLoadBalancer
from gateway.backends.base import BackendRegistry
from gateway.backends.openai_compatible import OpenAICompatibleBackend
from gateway.models.config_models import RoutesConfig, BackendConfig, ABTestConfig, ABTestVariant, TenantConfig

@pytest.fixture
def mock_registry():
    reg = BackendRegistry()
    b1 = OpenAICompatibleBackend(BackendConfig(name="b1", type="openai_compatible", base_url="x", models=["m1"]))
    b2 = OpenAICompatibleBackend(BackendConfig(name="b2", type="openai_compatible", base_url="x", models=["m1"]))
    b3 = OpenAICompatibleBackend(BackendConfig(name="b3", type="openai_compatible", base_url="x", models=["m2"]))
    reg.register(b1)
    reg.register(b2)
    reg.register(b3)
    reg.update_health("b1", True)
    reg.update_health("b2", True)
    reg.update_health("b3", False)
    return reg

def test_load_balancer(mock_registry):
    lb = WeightedLoadBalancer(mock_registry)
    
    # Empty
    assert lb.get_backend([]) is None
    
    # b3 is unhealthy, b1 and b2 are healthy
    backends = ["b1", "b2", "b3"]
    res1 = lb.get_backend(backends)
    res2 = lb.get_backend(backends)
    res3 = lb.get_backend(backends)
    
    assert res1.name in ("b1", "b2")
    assert res2.name in ("b1", "b2")
    assert res1.name != res2.name
    assert res3.name == res1.name

def test_fallback_chain(mock_registry):
    fb = FallbackChain(mock_registry, fallback_order=["b3", "b1", "b2"])
    
    # b3 is unhealthy, skips to b1
    res = fb.get_backend()
    assert res.name == "b1"
    
    tenant = TenantConfig(name="t1", api_key="k1", allowed_backends=["b2"])
    res_tenant = fb.get_backend(tenant)
    assert res_tenant.name == "b2"

def test_canary_router(mock_registry):
    canary = CanaryRouter(mock_registry)
    
    dep = CanaryDeployment(model="m1", stable_backend="b1", canary_backend="b2", traffic_percent=100)
    canary.add_deployment(dep)
    
    res = canary.route("m1")
    assert res.name == "b2"
    
    dep0 = CanaryDeployment(model="m2", stable_backend="b1", canary_backend="b2", traffic_percent=0)
    canary.add_deployment(dep0)
    assert canary.route("m2") is None
    
    # Rollback logic
    dep.record_failure()
    dep.record_failure()
    dep.record_failure()
    dep.record_failure()
    dep.record_failure() # 5 failures should trigger rollback if threshold is low
    
    # wait, default threshold is 0.1 (10%), 5 failures/5 req = 100% error rate
    assert dep.rolled_back is True
    # Once rolled back, returns None so it falls back to stable
    assert canary.route("m1") is None
    
    canary.remove_deployment("m1")
    assert canary.route("m1") is None

def test_ab_test_router(mock_registry):
    t1 = ABTestConfig(
        name="test1",
        enabled=True,
        model="m2",
        variants=[
            ABTestVariant(backend="b1", weight=0),
            ABTestVariant(backend="b3", weight=100) # b3 is unhealthy
        ]
    )
    ab = ABTestRouter(mock_registry, [t1])
    
    # b3 is unhealthy, ab router will skip and return None
    res = ab.route("m2", "tenant1", "req1")
    assert res is None

def test_request_router(mock_registry):
    cfg = RoutesConfig(
        routing={
            "default_backend": "b1",
            "fallback_order": ["b1", "b2"],
            "load_balancing": ["b1", "b2"]
        }
    )
    ab_test_router = ABTestRouter(mock_registry, [])
    canary_router = CanaryRouter(mock_registry)
    fallback_chain = FallbackChain(mock_registry, ["b1"])
    router = RequestRouter(mock_registry, cfg, ab_test_router, canary_router, fallback_chain)
    
    # Should resolve successfully (uses default/fallback)
    assert router.resolve("m1") is not None
    
    # Edge case: no healthy backend
    mock_registry.update_health("b1", False)
    mock_registry.update_health("b2", False)
    with pytest.raises(ValueError, match="No healthy backend found"):
        router.resolve("m1")
