import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner

from gateway.cli import app

runner = CliRunner()

def test_cli_serve_success():
    with patch("gateway.cli.uvicorn.run") as mock_run:
        result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000", "--workers", "2", "--reload", "--log-level", "debug"])
        assert result.exit_code == 0
        assert "127.0.0.1:9000" in result.stdout
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9000
        assert kwargs["workers"] == 2
        assert kwargs["reload"] is True
        assert kwargs["log_level"] == "debug"

def test_cli_serve_keyboard_interrupt():
    with patch("gateway.cli.uvicorn.run", side_effect=KeyboardInterrupt):
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "Gateway stopped" in result.stdout

def test_cli_serve_exception():
    with patch("gateway.cli.uvicorn.run", side_effect=RuntimeError("bind failed")):
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "Server failed: bind failed" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_health_success(mock_load, mock_get_settings):
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    
    # Mock routes config
    mock_load.return_value = {
        "routing": {"default_backend": "b1"},
        "backends": [
            {"name": "b1", "type": "openai_compatible", "base_url": "http://1", "models": ["m1"]},
            {"name": "b2", "type": "openai_compatible", "base_url": "http://2", "models": ["m1"]},
            {"name": "b3", "type": "openai_compatible", "base_url": "http://3", "models": ["m2"]},
            {"name": "b4", "type": "openai_compatible", "base_url": "http://4", "models": ["m2"]}
        ]
    }
    
    with patch("gateway.backends.base.BackendFactory.create") as mock_factory:
        mock_b1 = AsyncMock()
        mock_b1.name = "b1"
        mock_b1.backend_type = "openai_compatible"
        mock_b1._base_url = "http://1"
        mock_b1.health.return_value = True
        
        mock_b2 = AsyncMock()
        mock_b2.name = "b2"
        mock_b2.backend_type = "openai_compatible"
        mock_b2._base_url = "http://2"
        mock_b2.health.return_value = False
        
        mock_b3 = AsyncMock()
        mock_b3.name = "b3"
        mock_b3.backend_type = "openai_compatible"
        mock_b3._base_url = "http://3"
        mock_b3.health.side_effect = Exception("timeout")
        
        mock_factory.side_effect = [mock_b1, mock_b2, mock_b3, Exception("init failed")]
        
        # The CLI calls factory.create(), registers them, then iterates all_backends.
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        assert "Backend Health Check" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_health_config_error(mock_load, mock_get_settings):
    mock_load.side_effect = Exception("parse error")
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 1
    assert "Failed to load routes config: parse error" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_health_async_error(mock_load, mock_get_settings):
    mock_load.return_value = {"routing": {}, "backends": []}
    with patch("asyncio.run", side_effect=Exception("async crash")):
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 1
        assert "Health check failed: async crash" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_models(mock_load, mock_get_settings):
    mock_load.return_value = {
        "routing": {"default_backend": "b1"},
        "backends": [
            {"name": "b1", "type": "openai_compatible", "base_url": "http://1", "models": ["m1", "m2"]}
        ]
    }
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Configured Models" in result.output
    assert "b1 (openai_compatible):" in result.output
    assert "• m1" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_models_error(mock_load, mock_get_settings):
    mock_load.side_effect = Exception("error")
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "Failed to load config: error" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_config_validate_success(mock_load, mock_get_settings):
    def mock_load_yaml(path):
        if "routes" in path:
            return {"routing": {}, "backends": []}
        if "ab_tests" in path:
            return {"ab_tests": []}
        if "tenants" in path:
            return {"tenants": []}
        return {}

    mock_load.side_effect = mock_load_yaml
    mock_settings = MagicMock()
    mock_settings.resolve_config_path = lambda x: x
    mock_settings.routes_config = "routes.yaml"
    mock_settings.ab_tests_config = "ab_tests.yaml"
    mock_settings.tenants_config = "tenants.yaml"
    mock_get_settings.return_value = mock_settings

    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0
    assert "✅ All 3 configs valid!" in result.output

@patch("gateway.config.get_settings")
@patch("gateway.config.load_yaml_config")
def test_cli_config_validate_error(mock_load, mock_get_settings):
    mock_load.side_effect = Exception("invalid syntax")
    mock_settings = MagicMock()
    mock_settings.resolve_config_path = lambda x: x
    mock_settings.routes_config = "routes.yaml"
    mock_settings.ab_tests_config = "ab_tests.yaml"
    mock_settings.tenants_config = "tenants.yaml"
    mock_get_settings.return_value = mock_settings

    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 1
    assert "3 validation error" in result.output
    assert "invalid syntax" in result.output
