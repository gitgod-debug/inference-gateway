"""SSE (Server-Sent Events) response handler.

Bridges backend streaming responses to client SSE connections.
Handles chunk aggregation, [DONE] sentinel, and usage tracking.

Production considerations:
  - Client disconnect detection (don't keep sending to closed connections)
  - Backend errors during streaming record circuit breaker failures
  - Resource cleanup in finally block
  - Malformed chunk handling (log and skip, don't crash stream)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sse_starlette.sse import EventSourceResponse

from gateway.middleware.metrics import record_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request

    from gateway.models.response import ChatChunk
    from gateway.routing.fallback import FallbackChain

logger = logging.getLogger(__name__)


async def create_sse_response(
    request: Request,
    chunk_iterator: AsyncIterator[ChatChunk],
    model: str,
    backend_name: str,
    fallback_chain: FallbackChain | None = None,
) -> EventSourceResponse:
    """Create an SSE response from a backend chunk iterator.

    Wraps the async iterator to:
      1. Re-emit chunks in OpenAI SSE format
      2. Track token usage during streaming
      3. Send [DONE] sentinel at the end
      4. Handle client disconnects gracefully
      5. Record circuit breaker results
    """

    async def event_generator() -> AsyncIterator[dict]:
        total_completion_tokens = 0
        success = True
        try:
            async for chunk in chunk_iterator:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected during stream (backend=%s)", backend_name)
                    return

                # Count tokens from content deltas
                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        total_completion_tokens += max(1, len(choice.delta.content) // 4)

                try:
                    yield {
                        "event": "message",
                        "data": chunk.model_dump_json(exclude_none=True),
                    }
                except Exception as e:
                    logger.warning("Failed to serialize chunk from %s: %s", backend_name, e)
                    continue

            # Send [DONE] sentinel
            yield {"event": "message", "data": "[DONE]"}

        except GeneratorExit:
            # Client disconnected — normal, not an error
            logger.debug("SSE generator closed (client disconnect, backend=%s)", backend_name)
            return
        except Exception as e:
            success = False
            logger.error("SSE stream error from %s: %s", backend_name, e, exc_info=True)
            try:
                error_data = json.dumps({
                    "error": {"message": "Stream interrupted",
                              "type": "stream_error", "backend": backend_name}
                })
                yield {"event": "error", "data": error_data}
            except Exception:
                pass  # Can't send error to already-broken connection
        finally:
            # Record metrics and circuit breaker result
            if total_completion_tokens > 0:
                record_tokens(model=model, prompt_tokens=0,
                              completion_tokens=total_completion_tokens)
            if fallback_chain:
                fallback_chain.record_result(backend_name, success=success)

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )
