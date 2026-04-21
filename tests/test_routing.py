"""Tests for A/B test and routing logic."""


from gateway.backends.base import BackendRegistry
from gateway.backends.openai_compatible import OpenAICompatibleBackend
from gateway.models.config_models import ABTestConfig, ABTestVariant, BackendConfig
from gateway.routing.ab_test import ABTestRouter


def _make_backend(name: str) -> OpenAICompatibleBackend:
    config = BackendConfig(name=name, type="openai_compatible",
                           base_url="http://localhost:9999", models=["m1"])
    return OpenAICompatibleBackend(config)


class TestABTestRouter:
    """Test A/B test traffic splitting."""

    def _setup(self):
        registry = BackendRegistry()
        ba = _make_backend("backend-a")
        bb = _make_backend("backend-b")
        registry.register(ba)
        registry.register(bb)
        return registry

    def test_disabled_test_returns_none(self):
        registry = self._setup()
        test = ABTestConfig(
            name="test1", enabled=False, model="*",
            variants=[ABTestVariant(backend="backend-a", weight=50),
                      ABTestVariant(backend="backend-b", weight=50)],
        )
        router = ABTestRouter(registry=registry, tests=[test])
        assert router.route(model="m1") is None

    def test_weighted_distribution(self):
        registry = self._setup()
        test = ABTestConfig(
            name="test1", enabled=True, model="*",
            variants=[ABTestVariant(backend="backend-a", weight=70),
                      ABTestVariant(backend="backend-b", weight=30)],
        )
        router = ABTestRouter(registry=registry, tests=[test])

        counts = {"backend-a": 0, "backend-b": 0}
        for _ in range(1000):
            result = router.route(model="m1")
            if result:
                counts[result.name] += 1

        # With 70/30 split, backend-a should get ~700 (±50)
        assert 600 < counts["backend-a"] < 800
        assert 200 < counts["backend-b"] < 400

    def test_sticky_sessions(self):
        registry = self._setup()
        test = ABTestConfig(
            name="sticky", enabled=True, model="*", sticky=True,
            variants=[ABTestVariant(backend="backend-a", weight=50),
                      ABTestVariant(backend="backend-b", weight=50)],
        )
        router = ABTestRouter(registry=registry, tests=[test])

        # Same tenant should always get the same variant
        results = set()
        for _ in range(100):
            result = router.route(model="m1", tenant_name="tenant-1")
            if result:
                results.add(result.name)
        assert len(results) == 1  # Consistent hashing → always same backend

    def test_model_filtering(self):
        registry = self._setup()
        test = ABTestConfig(
            name="model-specific", enabled=True, model="special-model",
            variants=[ABTestVariant(backend="backend-a", weight=50),
                      ABTestVariant(backend="backend-b", weight=50)],
        )
        router = ABTestRouter(registry=registry, tests=[test])

        # Non-matching model should return None
        assert router.route(model="other-model") is None
