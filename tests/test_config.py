"""Tests for YAML config loading and validation."""


from gateway.config import _interpolate_env_vars, load_yaml_config, GatewaySettings, Environment, get_settings
from gateway.models.config_models import (
    BackendConfig,
    RoutesConfig,
    TenantConfig,
)
from gateway.models.request import ChatRequest
import os


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

    def test_interpolate_other_types(self):
        assert _interpolate_env_vars(123) == 123
        assert _interpolate_env_vars(None) is None

    def test_load_yaml_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        data = load_yaml_config(cfg)
        assert data == {}

    def test_load_yaml_not_a_dict(self, tmp_path):
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- a\n- b")
        data = load_yaml_config(cfg)
        assert data == {}

    def test_load_yaml_invalid_syntax(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("key: value\n  broken_indent: oops")
        data = load_yaml_config(cfg)
        assert data == {}

    def test_load_yaml_os_error(self, tmp_path, monkeypatch):
        # Create a directory with the same name, so trying to open it as a file fails
        cfg = tmp_path / "dir.yaml"
        cfg.mkdir()
        data = load_yaml_config(cfg)
        assert data == {}

class TestGatewaySettings:
    def test_log_level_normalise(self):
        s = GatewaySettings(log_level="DEBUG")
        assert s.log_level == "debug"

    def test_is_production(self):
        s = GatewaySettings(env="production")
        assert s.is_production is True
        s2 = GatewaySettings(env="development")
        assert s2.is_production is False

    def test_resolve_config_path(self):
        s = GatewaySettings()
        # Relative
        path = s.resolve_config_path("test.yaml")
        assert path.is_absolute()
        assert path.name == "test.yaml"
        # Absolute
        abs_path = os.path.abspath("/tmp/test.yaml")
        path2 = s.resolve_config_path(abs_path)
        assert str(path2) == abs_path

    def test_get_settings_cache(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2



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

    def test_invalid_allowed_backends(self):
        t = TenantConfig(name="t1", api_key="k1", allowed_backends="not_a_list")
        # Should return False because it's neither '*' nor a list
        assert t.can_access_backend("b1") is False

class TestChatRequest:
    def test_to_backend_payload(self):
        req = ChatRequest(model="test", messages=[{"role": "user", "content": "hi"}], temperature=0.5)
        payload = req.to_backend_payload()
        assert "model" in payload
        assert "temperature" in payload
        # 'seed' is None by default, should be excluded
        assert "seed" not in payload
