"""Tests for SSE streaming."""

from gateway.models.response import ChatChunk, ChatChunkChoice, ChatChunkDelta


class TestSSEChunkFormat:

    def test_chunk_serializes(self):
        chunk = ChatChunk(id="c-1", model="test",
            choices=[ChatChunkChoice(delta=ChatChunkDelta(content="Hello"))])
        data = chunk.model_dump_json(exclude_none=True)
        assert "Hello" in data

    def test_finish_reason(self):
        chunk = ChatChunk(id="c-1", model="test",
            choices=[ChatChunkChoice(delta=ChatChunkDelta(), finish_reason="stop")])
        d = chunk.model_dump(exclude_none=True)
        assert d["choices"][0]["finish_reason"] == "stop"

    def test_role_delta(self):
        chunk = ChatChunk(id="c-1", model="test",
            choices=[ChatChunkChoice(delta=ChatChunkDelta(role="assistant"))])
        d = chunk.model_dump(exclude_none=True)
        assert d["choices"][0]["delta"]["role"] == "assistant"
