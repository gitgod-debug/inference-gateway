"""Ollama backend adapter.

Ollama provides both a native API (/api/chat) and an OpenAI-compatible
endpoint (/v1/chat/completions). We use the OpenAI-compatible endpoint
for consistency, with the native API used for health checks and model
discovery.

Reference: https://github.com/ollama/ollama/blob/main/docs/openai.md
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

    from gateway.models.config_models import BackendConfig
    from gateway.models.request import ChatRequest

logger = logging.getLogger(__name__)


@BackendFactory.register("ollama")
class OllamaBackend(BaseHTTPBackend):
    """Backend adapter for Ollama inference server.

    Uses OpenAI-compatible /v1/chat/completions for inference,
    and native /api/tags for health and model discovery.
    """

    def __init__(self, config: BackendConfig) -> None:
        # Ollama doesn't require an API key
        super().__init__(config)
        self._completions_path = "/v1/chat/completions"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming completion via Ollama's OpenAI-compat endpoint."""
        payload = request.to_backend_payload()
        payload["stream"] = False

        try:
            resp = await self._client.post(self._completions_path, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)
        except httpx.TimeoutException as e:
            logger.error("Ollama %s timed out: %s", self._name, e)
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                "Ollama %s returned %d: %s",
                self._name, e.response.status_code, e.response.text[:500],
            )
            raise
        except httpx.HTTPError as e:
            logger.error("Ollama %s HTTP error: %s", self._name, e)
            raise

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Stream completions from Ollama's OpenAI-compat endpoint."""
        payload = request.to_backend_payload()
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST", self._completions_path, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            yield self._parse_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as e:
            logger.error("Ollama %s stream error: %s", self._name, e)
            raise

    async def health(self) -> bool:
        """Check Ollama health via native /api/tags endpoint."""
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.HTTPError, Exception):
            return False

    async def list_models(self) -> list[str]:
        """Discover available models from Ollama's /api/tags endpoint."""
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except (httpx.HTTPError, Exception):
            pass
        return self._models

    def _parse_response(self, data: dict) -> ChatResponse:
        """Parse OpenAI-format response from Ollama."""
        try:
            choices = []
            for choice_data in data.get("choices", []):
                msg = choice_data.get("message", {}) if isinstance(choice_data, dict) else {}
                choices.append(
                    ChatChoice(
                        index=choice_data.get("index", 0) if isinstance(choice_data, dict) else 0,
                        message=ChatChoiceMessage(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content"),
                        ),
                        finish_reason=choice_data.get("finish_reason") if isinstance(choice_data, dict) else None,
                    )
                )

            usage_data = data.get("usage") or {}
            return ChatResponse(
                id=data.get("id", ""),
                model=data.get("model", ""),
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                ),
                backend=self._name,
            )
        except Exception as e:
            logger.error("Ollama %s parse error: %s", self._name, e)
            return ChatResponse(model=data.get("model", ""), backend=self._name)

    def _parse_chunk(self, data: dict) -> ChatChunk:
        """Parse SSE chunk from Ollama."""
        try:
            choices = []
            for choice_data in data.get("choices", []):
                delta = choice_data.get("delta", {}) if isinstance(choice_data, dict) else {}
                choices.append(
                    ChatChunkChoice(
                        index=choice_data.get("index", 0) if isinstance(choice_data, dict) else 0,
                        delta=ChatChunkDelta(
                            role=delta.get("role"),
                            content=delta.get("content"),
                        ),
                        finish_reason=choice_data.get("finish_reason") if isinstance(choice_data, dict) else None,
                    )
                )
            return ChatChunk(
                id=data.get("id", ""),
                model=data.get("model", ""),
                choices=choices,
                backend=self._name,
            )
        except Exception as e:
            logger.error("Ollama %s chunk parse error: %s", self._name, e)
            return ChatChunk(model=data.get("model", ""), backend=self._name)
