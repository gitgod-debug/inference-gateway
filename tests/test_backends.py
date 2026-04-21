"""Tests for backend adapters using mocked HTTP responses."""

import pytest
import respx
from httpx import Response

from gateway.backends.ollama import OllamaBackend
from gateway.backends.openai_compatible import OpenAICompatibleBackend
from gateway.models.config_models import BackendConfig
from gateway.models.request import ChatMessage, ChatRequest


def _make_request() -> ChatRequest:
    return ChatRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
    )


def _mock_response() -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class TestOpenAICompatibleBackend:

    @pytest.mark.asyncio
    @respx.mock
    async def test_complete(self):
        cfg = BackendConfig(name="test", type="openai_compatible",
                           base_url="http://mock-api.test", models=["m1"])
        backend = OpenAICompatibleBackend(cfg)

        respx.post("http://mock-api.test/chat/completions").mock(
            return_value=Response(200, json=_mock_response())
        )

        resp = await backend.complete(_make_request())
        assert resp.choices[0].message.content == "Hi there!"
        assert resp.usage.total_tokens == 8
        assert resp.backend == "test"
        await backend.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check(self):
        cfg = BackendConfig(name="test", type="openai_compatible",
                           base_url="http://mock-api.test", models=["m1"],
                           health_endpoint="/health")
        backend = OpenAICompatibleBackend(cfg)

        respx.get("http://mock-api.test/health").mock(
            return_value=Response(200)
        )

        assert await backend.health() is True
        await backend.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_failure(self):
        cfg = BackendConfig(name="test", type="openai_compatible",
                           base_url="http://mock-api.test", models=["m1"])
        backend = OpenAICompatibleBackend(cfg)

        respx.get("http://mock-api.test/health").mock(
            return_value=Response(500)
        )

        assert await backend.health() is False
        await backend.close()


class TestOllamaBackend:

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_models(self):
        cfg = BackendConfig(name="ollama", type="ollama",
                           base_url="http://mock-ollama.test", models=[])
        backend = OllamaBackend(cfg)

        respx.get("http://mock-ollama.test/api/tags").mock(
            return_value=Response(200, json={
                "models": [{"name": "llama3.2"}, {"name": "mistral"}]
            })
        )

        models = await backend.list_models()
        assert "llama3.2" in models
        assert "mistral" in models
        await backend.close()
