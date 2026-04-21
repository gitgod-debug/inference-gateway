"""Tests for YAML config loading and validation."""


from gateway.config import _interpolate_env_vars, load_yaml_config
from gateway.models.config_models import (
    BackendConfig,
    RoutesConfig,
    TenantConfig,
)


class TestConfigLoading:

    def test_load_yaml_file(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("key: value\nlist:\n  - a\n  - b")
        data = load_yaml_config(cfg)
        assert data["key"] == "value"
        assert data["list"] == ["a", "b"]

    def test_missing_file_returns_empty(self, tmp_path):
        data = load_yaml_config(tmp_path / "missing.yaml")
        assert data == {}

    def test_env_var_interpolation(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello-world")
        result = _interpolate_env_vars("prefix-${TEST_VAR}-suffix")
        assert result == "prefix-hello-world-suffix"

    def test_nested_interpolation(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret123")
        data = {"backends": [{"api_key": "${API_KEY}"}]}
        result = _interpolate_env_vars(data)
        assert result["backends"][0]["api_key"] == "secret123"

    def test_missing_env_var_becomes_empty(self):
        result = _interpolate_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert result == ""


class TestBackendConfig:

    def test_valid_backend(self):
        cfg = BackendConfig(
            name="test", type="openai_compatible",
            base_url="http://localhost:8000", models=["m1"],
        )
        assert cfg.name == "test"
        assert cfg.priority == 10  # default

    def test_all_backend_types(self):
        for t in ["ollama", "vllm", "sglang", "openai_compatible"]:
            cfg = BackendConfig(name=f"b-{t}", type=t, base_url="http://x")
            assert cfg.type == t


class TestRoutesConfig:

    def test_valid_routes(self):
        cfg = RoutesConfig(
            backends=[BackendConfig(name="b1", type="vllm", base_url="http://x")],
            routing={"default_backend": "b1", "fallback_order": ["b1"]},
        )
        assert len(cfg.backends) == 1
        assert cfg.routing.default_backend == "b1"

    def test_empty_routes(self):
        cfg = RoutesConfig()
        assert len(cfg.backends) == 0


class TestTenantConfig:

    def test_wildcard_access(self):
        t = TenantConfig(name="t1", api_key="k1", allowed_backends="*")
        assert t.can_access_backend("any-backend") is True

    def test_list_access(self):
        t = TenantConfig(name="t1", api_key="k1", allowed_backends=["b1", "b2"])
        assert t.can_access_backend("b1") is True
        assert t.can_access_backend("b3") is False

    def test_tier_validation(self):
        t = TenantConfig(name="t1", api_key="k1", tier="enterprise")
        assert t.tier == "enterprise"
