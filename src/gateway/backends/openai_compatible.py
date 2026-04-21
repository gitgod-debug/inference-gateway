"""OpenAI-compatible backend adapter.

This is the most versatile adapter — it works with any API that implements
the OpenAI Chat Completions spec:
  - Groq          (https://api.groq.com/openai/v1)
  - OpenRouter    (https://openrouter.ai/api/v1)
  - Google Gemini (https://generativelanguage.googleapis.com/v1beta/openai/)
  - OpenAI        (https://api.openai.com/v1)
  - Together AI   (https://api.together.xyz/v1)
  - Any BYOK provider with OpenAI-compatible API

All three free backends use this adapter.
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


@BackendFactory.register("openai_compatible")
class OpenAICompatibleBackend(BaseHTTPBackend):
    """Backend adapter for any OpenAI-compatible API.

    Handles both streaming (SSE) and non-streaming requests.
    Used by Groq, OpenRouter, Gemini, and any BYOK provider.
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._completions_path = "/chat/completions"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request."""
        payload = request.to_backend_payload()
        payload["stream"] = False

        try:
            resp = await self._client.post(self._completions_path, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)
        except httpx.TimeoutException as e:
            logger.error("Backend %s timed out: %s", self._name, e)
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                "Backend %s returned %d: %s",
                self._name, e.response.status_code, e.response.text[:500],
            )
            raise
        except httpx.HTTPError as e:
            logger.error("Backend %s HTTP error: %s", self._name, e)
            raise

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Send a streaming chat completion request, yielding SSE chunks."""
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
                    # SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            yield self._parse_chunk(data)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Backend %s sent malformed chunk: %s",
                                self._name, data_str[:200],
                            )
                            continue
        except httpx.HTTPStatusError as e:
            try:
                await e.response.aread()
                body = e.response.text[:500]
            except Exception:
                body = "<unreadable>"
            logger.error(
                "Backend %s stream error %d: %s",
                self._name, e.response.status_code, body,
            )
            raise
        except httpx.HTTPError as e:
            logger.error("Backend %s stream HTTP error: %s", self._name, e)
            raise

    def _parse_response(self, data: dict) -> ChatResponse:
        """Parse a raw JSON response into a ChatResponse.

        Handles malformed responses gracefully — returns empty choices
        rather than crashing on unexpected data shapes.
        """
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
            usage = UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return ChatResponse(
                id=data.get("id", ""),
                model=data.get("model", ""),
                choices=choices,
                usage=usage,
                backend=self._name,
            )
        except Exception as e:
            logger.error("Failed to parse response from %s: %s (data=%s)",
                        self._name, e, str(data)[:500])
            # Return empty response rather than crashing
            return ChatResponse(model=data.get("model", ""), backend=self._name)

    def _parse_chunk(self, data: dict) -> ChatChunk:
        """Parse a raw SSE JSON chunk into a ChatChunk.

        Handles malformed chunks gracefully.
        """
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
            logger.error("Failed to parse chunk from %s: %s", self._name, e)
            return ChatChunk(model=data.get("model", ""), backend=self._name)
