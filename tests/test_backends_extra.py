import pytest
import respx
import httpx
from httpx import Response
from unittest.mock import patch
import asyncio

from gateway.backends.base import BackendFactory, BackendRegistry, BaseHTTPBackend
from gateway.models.config_models import BackendConfig
from gateway.models.request import ChatRequest
from gateway.backends.openai_compatible import OpenAICompatibleBackend
from gateway.backends.ollama import OllamaBackend
from gateway.backends.vllm import VLLMBackend
from gateway.backends.sglang import SGLangBackend

# --- base.py: Factory & Registry ---
@pytest.mark.asyncio
async def test_backend_factory_and_registry():
    cfg_oai = BackendConfig(name="b1", type="openai_compatible", base_url="http://b1")
    cfg_bad = BackendConfig.model_construct(name="b2", type="unknown_type", base_url="http://b2")
    
    with pytest.raises(ValueError, match="Unknown backend type"):
        BackendFactory.create(cfg_bad)
    
    b1 = BackendFactory.create(cfg_oai)
    assert isinstance(b1, OpenAICompatibleBackend)
    assert b1.name == "b1"
    
    # Registry
    reg = BackendRegistry()
    reg.register(b1)
    
    assert reg.get("b1") is b1
    assert reg.get("nope") is None
    
    # Health tracking
    assert reg.is_healthy("b1") is True
    reg.update_health("b1", False)
    assert reg.is_healthy("b1") is False
    reg.update_health("b1", False) # same state, covers `if current_health == is_healthy: return`
    
    assert reg.get_healthy_backends() == []
    reg.update_health("b1", True)
    assert len(reg.get_healthy_backends()) == 1
    
    assert reg.all_names == ["b1"]
    status = reg.health_status
    assert status["b1"] is True
    
    await reg.close_all()

@pytest.mark.asyncio
async def test_base_backend_edge_cases():
    cfg = BackendConfig(name="b1", type="openai_compatible", base_url="http://b1/", api_key="testkey", models=["m1"])
    b = BackendFactory.create(cfg)
    
    assert b._base_url == "http://b1"  # stripped trailing slash
    assert "Authorization" in b._client.headers
    assert repr(b).startswith("<OpenAICompatibleBackend")
    assert await b.list_models() == ["m1"]
    
    with respx.mock:
        respx.get("http://b1/health").mock(side_effect=httpx.TimeoutException("timeout"))
        assert await b.health() is False
    await b.close()

# --- Common Helper ---
def _mock_req():
    return ChatRequest(model="m1", messages=[{"role": "user", "content": "hi"}])

# --- OpenAI Compatible ---
@pytest.mark.asyncio
async def test_openai_compatible_edge_cases():
    cfg = BackendConfig(name="oai", type="openai_compatible", base_url="http://oai")
    b = OpenAICompatibleBackend(cfg)
    req = _mock_req()
    
    with respx.mock:
        # complete: timeout error
        respx.post("http://oai/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            await b.complete(req)
            
        # complete: 500 error
        respx.post("http://oai/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            await b.complete(req)
            
        # complete: malformed JSON parsing
        respx.post("http://oai/chat/completions").mock(return_value=Response(200, json={"choices": [{"not_dict": True}]}))
        resp = await b.complete(req)
        assert resp.choices[0].message.content is None # default fallback
        
        # stream: errors
        respx.post("http://oai/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://oai/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in b.stream(req):
                pass
                
        # complete: general http error
        respx.post("http://oai/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            await b.complete(req)

        # stream: general http error
        respx.post("http://oai/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            async for _ in b.stream(req):
                pass

        # stream: bad chunks skipped
        async def mock_stream_response():
            yield b"data: not json\n\n"
            yield b"data: [DONE]\n\n"
        
        from httpx import AsyncByteStream
        class FakeStream(AsyncByteStream):
            async def __aiter__(self):
                async for chunk in mock_stream_response():
                    yield chunk
        respx.post("http://oai/chat/completions").mock(return_value=Response(200, stream=FakeStream()))
        chunks = [c async for c in b.stream(req)]
        assert len(chunks) == 0  # Invalid JSON skipped

    await b.close()

# --- Ollama ---
@pytest.mark.asyncio
async def test_ollama_edge_cases():
    cfg = BackendConfig(name="ollama", type="ollama", base_url="http://ollama")
    b = OllamaBackend(cfg)
    req = _mock_req()
    
    with respx.mock:
        # complete: timeout error
        respx.post("http://ollama/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            await b.complete(req)
            
        # complete: 500 error
        respx.post("http://ollama/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            await b.complete(req)
            
        # complete: general http error
        respx.post("http://ollama/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            await b.complete(req)
            
        # stream: errors
        respx.post("http://ollama/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://ollama/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://ollama/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            async for _ in b.stream(req):
                pass

        respx.get("http://ollama/api/tags").mock(return_value=Response(500))
        assert await b.health() is False
        assert await b.list_models() == [] # fallback to self._models which is []
        
        respx.post("http://ollama/v1/chat/completions").mock(return_value=Response(200, json={"choices":[{"message":{"content":"hi","role":"assistant"}}]}))
        resp = await b.complete(req)
        assert resp.choices[0].message.content == "hi"
        
        # streaming via compatibility
        async def mock_stream():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        from httpx import AsyncByteStream
        class FakeStream(AsyncByteStream):
            async def __aiter__(self):
                async for chunk in mock_stream():
                    yield chunk
        respx.post("http://ollama/v1/chat/completions").mock(return_value=Response(200, stream=FakeStream()))
        chunks = [c async for c in b.stream(req)]
        assert len(chunks) == 1
    
    await b.close()

# --- vLLM ---
@pytest.mark.asyncio
async def test_vllm_edge_cases():
    cfg = BackendConfig(name="vllm", type="vllm", base_url="http://vllm", models=["default"])
    b = VLLMBackend(cfg)
    req = _mock_req()
    
    with respx.mock:
        respx.get("http://vllm/health").mock(return_value=Response(200))
        assert await b.health() is True
        
        respx.get("http://vllm/health").mock(return_value=Response(500))
        assert await b.health() is False
        
        respx.get("http://vllm/v1/models").mock(return_value=Response(200, json={"data": [{"id": "m1"}]}))
        assert await b.list_models() == ["m1"]
        
        respx.get("http://vllm/v1/models").mock(side_effect=httpx.TimeoutException("t"))
        assert await b.list_models() == ["default"]
        
        # vLLM complete/stream inherits or reimplements OpenAI format
        respx.post("http://vllm/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            await b.complete(req)
            
        respx.post("http://vllm/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            await b.complete(req)
            
        respx.post("http://vllm/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            await b.complete(req)

        respx.post("http://vllm/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://vllm/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://vllm/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            async for _ in b.stream(req):
                pass

        respx.post("http://vllm/v1/chat/completions").mock(return_value=Response(200, json={"choices":[{"message":{"content":"hi"}}]  }))
        resp = await b.complete(req)
        assert resp.choices[0].message.content == "hi"

        async def mock_stream():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        from httpx import AsyncByteStream
        class FakeStream(AsyncByteStream):
            async def __aiter__(self):
                async for chunk in mock_stream():
                    yield chunk
        respx.post("http://vllm/v1/chat/completions").mock(return_value=Response(200, stream=FakeStream()))
        chunks = [c async for c in b.stream(req)]
        assert len(chunks) == 1

    await b.close()

# --- SGLang ---
@pytest.mark.asyncio
async def test_sglang_edge_cases():
    cfg = BackendConfig(name="sglang", type="sglang", base_url="http://sglang", models=["default"])
    b = SGLangBackend(cfg)
    req = _mock_req()
    
    with respx.mock:
        respx.get("http://sglang/health").mock(return_value=Response(200))
        assert await b.health() is True
        
        respx.get("http://sglang/health").mock(side_effect=httpx.TimeoutException("t"))
        assert await b.health() is False
        
        respx.get("http://sglang/v1/models").mock(return_value=Response(200, json={"data": [{"id": "m1"}]}))
        assert await b.list_models() == ["m1"]
        
        respx.get("http://sglang/v1/models").mock(return_value=Response(500))
        assert await b.list_models() == ["default"]
        
        respx.post("http://sglang/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            await b.complete(req)
            
        respx.post("http://sglang/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            await b.complete(req)
            
        respx.post("http://sglang/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            await b.complete(req)

        respx.post("http://sglang/v1/chat/completions").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(httpx.TimeoutException):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://sglang/v1/chat/completions").mock(return_value=Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in b.stream(req):
                pass
                
        respx.post("http://sglang/v1/chat/completions").mock(side_effect=httpx.NetworkError("network"))
        with pytest.raises(httpx.NetworkError):
            async for _ in b.stream(req):
                pass

        respx.post("http://sglang/v1/chat/completions").mock(return_value=Response(200, json={"choices":[{"message":{"content":"hi"}}]  }))
        resp = await b.complete(req)
        assert resp.choices[0].message.content == "hi"

        async def mock_stream():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        from httpx import AsyncByteStream
        class FakeStream(AsyncByteStream):
            async def __aiter__(self):
                async for chunk in mock_stream():
                    yield chunk
        respx.post("http://sglang/v1/chat/completions").mock(return_value=Response(200, stream=FakeStream()))
        chunks = [c async for c in b.stream(req)]
        assert len(chunks) == 1

    await b.close()
