import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager

from fastapi import FastAPI
from gateway.app import create_app, lifespan

@pytest.fixture
def mock_app():
    app = FastAPI()
    mock_settings = MagicMock()
    app.state.settings = mock_settings
    return app

@pytest.mark.asyncio
async def test_lifespan_success(mock_app):
    mock_app.state.settings.resolve_config_path.return_value = "config.yaml"
    
    with patch("gateway.app.load_yaml_config") as mock_load:
        # valid empty configs
        mock_load.return_value = {}
        
        async with lifespan(mock_app):
            assert mock_app.state.backend_registry is not None
            assert mock_app.state.request_router is not None
            assert mock_app.state.health_checker is not None
            
        # Assert cleanup
        assert mock_app.state.health_checker._running is False

@pytest.mark.asyncio
async def test_lifespan_config_errors(mock_app):
    mock_app.state.settings.resolve_config_path.return_value = "config.yaml"
    
    # Force load_yaml_config to throw to hit 74, 81, 88
    with patch("gateway.app.load_yaml_config", side_effect=Exception("parse error")):
        async with lifespan(mock_app):
            # It should fall back to empty defaults
            assert mock_app.state.backend_registry is not None
            assert mock_app.state.request_router is not None

@pytest.mark.asyncio
async def test_lifespan_backend_init_error(mock_app):
    mock_app.state.settings.resolve_config_path.return_value = "config.yaml"
    
    with patch("gateway.app.load_yaml_config") as mock_load:
        def mock_load_impl(path):
            if "routes" in path:
                return {"backends": [{"name": "bad", "type": "openai_compatible", "base_url": "http://x", "models": ["x"]}], "routing": {"default_backend": "bad"}}
            return {}
        mock_load.side_effect = mock_load_impl
        
        with patch("gateway.app.BackendFactory.create", side_effect=Exception("init error")):
            async with lifespan(mock_app):
                # Should catch error at 97 and log warning at 103
                assert len(mock_app.state.backend_registry.all_names) == 0

@pytest.mark.asyncio
async def test_lifespan_catastrophic_error(mock_app):
    mock_app.state.settings.resolve_config_path.return_value = "config.yaml"
    
    with patch("gateway.app.load_yaml_config", return_value={}):
        # Mocking CanaryRouter to crash to simulate catastrophic failure at 112
        with patch("gateway.app.CanaryRouter", side_effect=Exception("catastrophic")):
            async with lifespan(mock_app):
                # Caught at 149
                assert mock_app.state.request_router is None
                assert mock_app.state.fallback_chain is None

@pytest.mark.asyncio
async def test_lifespan_shutdown_error(mock_app):
    with patch("gateway.app.load_yaml_config", return_value={}):
        with patch("gateway.app.BackendRegistry.close_all", side_effect=Exception("shutdown error")):
            # Caught at 165
            async with lifespan(mock_app):
                pass
            # Does not raise, catches and logs

def test_create_app_function():
    with patch("gateway.app.load_yaml_config", side_effect=Exception("tenant load error")):
        with patch("gateway.app.Instrumentator") as mock_inst:
            mock_inst.side_effect = Exception("prometheus error")
            app = create_app()
            
            assert isinstance(app, FastAPI)
            assert app.title == "Inference Gateway"
            assert len(app.user_middleware) > 0
