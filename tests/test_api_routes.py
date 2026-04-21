import pytest
from httpx import AsyncClient, Request, Response, HTTPStatusError, TimeoutException
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from httpx import ASGITransport

from gateway.api.routes import router
from gateway.models.request import ChatRequest
from gateway.models.response import ChatResponse, UsageInfo

@pytest.fixture
def mock_app():
    app = FastAPI()
    app.include_router(router)
    
    app.state.request_router = MagicMock()
    app.state.fallback_chain = MagicMock()
    
    return app

@pytest.fixture
async def client(mock_app):
    async with AsyncClient(transport=ASGITransport(app=mock_app), base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_chat_completions_success(client, mock_app):
    mock_backend = AsyncMock()
    mock_backend.name = "test_backend"
    
    mock_resp = ChatResponse(id="1", object="chat.completion", created=123, model="m1", choices=[], usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30))
    mock_backend.complete.return_value = mock_resp
    
    mock_app.state.request_router.resolve.return_value = mock_backend
    
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["id"] == "1"
    
    mock_app.state.fallback_chain.record_result.assert_called_once_with("test_backend", success=True)

@pytest.mark.asyncio
async def test_chat_completions_streaming(client, mock_app):
    mock_backend = AsyncMock()
    mock_backend.name = "test_backend"
    
    async def mock_stream(*args, **kwargs):
        yield None
    
    mock_backend.stream = mock_stream
    mock_app.state.request_router.resolve.return_value = mock_backend
    
    # We won't fully consume the stream or test sse logic here (it's covered in streaming tests),
    # but we can hit the endpoint to cover lines 80-87
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_chat_completions_no_backend(client, mock_app):
    mock_app.state.request_router.resolve.side_effect = ValueError("No backend")
    
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_backend"

@pytest.mark.asyncio
async def test_chat_completions_http_error(client, mock_app):
    mock_backend = AsyncMock()
    mock_backend.name = "test_backend"
    
    req = Request("POST", "http://test")
    resp = Response(500, request=req, text="internal error")
    mock_backend.complete.side_effect = HTTPStatusError("error", request=req, response=resp)
    
    mock_app.state.request_router.resolve.return_value = mock_backend
    
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_http_error"
    mock_app.state.fallback_chain.record_result.assert_called_once_with("test_backend", success=False)

@pytest.mark.asyncio
async def test_chat_completions_timeout(client, mock_app):
    mock_backend = AsyncMock()
    mock_backend.name = "test_backend"
    
    mock_backend.complete.side_effect = TimeoutException("timeout")
    mock_app.state.request_router.resolve.return_value = mock_backend
    
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "backend_timeout"
    mock_app.state.fallback_chain.record_result.assert_called_once_with("test_backend", success=False)

@pytest.mark.asyncio
async def test_chat_completions_generic_error(client, mock_app):
    mock_backend = AsyncMock()
    mock_backend.name = "test_backend"
    
    mock_backend.complete.side_effect = Exception("generic error")
    mock_app.state.request_router.resolve.return_value = mock_backend
    
    response = await client.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "unexpected_error"
    mock_app.state.fallback_chain.record_result.assert_called_once_with("test_backend", success=False)
