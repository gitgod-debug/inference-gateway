"""OpenAI-compatible request schemas.

These models define the request format accepted by the gateway.
They follow the OpenAI Chat Completions API specification so that
any client built for OpenAI can use the gateway without changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Size limits to prevent memory exhaustion ────────────────
_MAX_MESSAGES = 100        # Max messages per request
_MAX_CONTENT_CHARS = 32768 # Max chars per message (≈8K tokens)
_MAX_STOP_SEQUENCES = 4    # Max stop sequences


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message author"
    )
    content: str | None = Field(
        default=None, max_length=_MAX_CONTENT_CHARS,
        description="The message content",
    )
    name: str | None = Field(
        default=None, max_length=64,
        description="Optional name for the participant",
    )


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str = Field(..., max_length=256, description="Model identifier to use for completion")
    messages: list[ChatMessage] = Field(
        ..., min_length=1, max_length=_MAX_MESSAGES,
        description="List of messages in the conversation",
    )

    # ── Generation parameters ───────────────────────────────
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, le=16384, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Enable SSE streaming")
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences (max 4)"
    )
    n: int = Field(default=1, ge=1, le=5, description="Number of completions")

    # ── Advanced parameters ─────────────────────────────────
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    user: str | None = Field(default=None, description="End-user identifier")

    def to_backend_payload(self) -> dict:
        """Convert to a dict suitable for forwarding to a backend.

        Excludes None values to avoid overriding backend defaults.
        """
        return self.model_dump(exclude_none=True)
