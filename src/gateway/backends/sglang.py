"""SGLang backend adapter.

SGLang exposes an OpenAI-compatible API. This adapter adds SGLang-specific
health checking. Reference: https://github.com/sgl-project/sglang
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

from gateway.backends.base import BackendFactory, BaseHTTPBackend
from gateway.models.response import (
    ChatChoice,
    ChatChoiceMessage,
    ChatChunk,
    ChatChunkChoice,
    ChatChunkDelta,
    ChatResponse,
    UsageInfo,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gateway.models.request import ChatRequest

logger = logging.getLogger(__name__)


@BackendFactory.register("sglang")
class SGLangBackend(BaseHTTPBackend):
    """Backend adapter for SGLang inference server."""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = request.to_backend_payload()
        payload["stream"] = False
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        payload = request.to_backend_payload()
        payload["stream"] = True
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    yield self._parse_chunk(json.loads(data_str))
                except json.JSONDecodeError:
                    continue

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except (httpx.HTTPError, Exception):
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            if resp.status_code == 200:
                return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, Exception):
            pass
        return self._models

    def _parse_response(self, data: dict) -> ChatResponse:
        choices = [
            ChatChoice(
                index=c.get("index", 0),
                message=ChatChoiceMessage(content=c.get("message", {}).get("content")),
                finish_reason=c.get("finish_reason"),
            ) for c in data.get("choices", [])
        ]
        u = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""), model=data.get("model", ""), choices=choices,
            usage=UsageInfo(prompt_tokens=u.get("prompt_tokens", 0),
                           completion_tokens=u.get("completion_tokens", 0),
                           total_tokens=u.get("total_tokens", 0)),
            backend=self._name,
        )

    def _parse_chunk(self, data: dict) -> ChatChunk:
        choices = [
            ChatChunkChoice(
                index=c.get("index", 0),
                delta=ChatChunkDelta(role=c.get("delta", {}).get("role"),
                                     content=c.get("delta", {}).get("content")),
                finish_reason=c.get("finish_reason"),
            ) for c in data.get("choices", [])
        ]
        return ChatChunk(id=data.get("id", ""), model=data.get("model", ""),
                         choices=choices, backend=self._name)
