"""OpenAI-compatible response schemas.

These models define the response format returned by the gateway.
Both streaming (SSE chunks) and non-streaming responses follow
the OpenAI Chat Completions API specification.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ChatChoiceMessage(BaseModel):
    """The message content of a chat choice."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChatChoiceMessage = Field(default_factory=ChatChoiceMessage)
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response (non-streaming).

    Reference: https://platform.openai.com/docs/api-reference/chat/object
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)

    # ── Gateway-specific metadata (not in OpenAI spec) ──────
    backend: str | None = Field(
        default=None,
        description="Which backend served this response",
        exclude=True,
    )


# ── Streaming (SSE) models ──────────────────────────────────


class ChatChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatChunkDelta = Field(default_factory=ChatChunkDelta)
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    """OpenAI-compatible streaming chunk.

    Reference: https://platform.openai.com/docs/api-reference/chat/streaming
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatChunkChoice] = Field(default_factory=list)

    # ── Gateway-specific metadata ───────────────────────────
    backend: str | None = Field(default=None, exclude=True)


# ── Error response ──────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Structured error detail."""

    message: str
    type: str = "gateway_error"
    code: str | None = None
    backend: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: ErrorDetail


# ── Model listing ───────────────────────────────────────────


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "inference-gateway"
    backend: str | None = None


class ModelListResponse(BaseModel):
    """Response for GET /v1/models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(default_factory=list)
