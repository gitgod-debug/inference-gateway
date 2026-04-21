import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request

from gateway.streaming.sse import create_sse_response
from gateway.models.response import ChatChunk, ChatChunkChoice, ChatChunkDelta
from sse_starlette.sse import EventSourceResponse

class DummyRequest:
    def __init__(self, disconnected=False):
        self._disconnected = disconnected
    async def is_disconnected(self):
        return self._disconnected

@pytest.mark.asyncio
async def test_sse_streamer_normal():
    # Setup mock chunks
    chunk1 = ChatChunk(id="1", model="m1", backend="b1", choices=[
        ChatChunkChoice(index=0, delta=ChatChunkDelta(role="assistant", content="hello "))
    ])
    chunk2 = ChatChunk(id="1", model="m1", backend="b1", choices=[
        ChatChunkChoice(index=0, delta=ChatChunkDelta(content="world"))
    ])
    
    async def mock_iterator():
        yield chunk1
        yield chunk2
        
    fallback = MagicMock()
    req = DummyRequest()
    
    resp = await create_sse_response(
        chunk_iterator=mock_iterator(),
        backend_name="b1",
        model="m1",
        request=req,
        fallback_chain=fallback
    )
    
    assert isinstance(resp, EventSourceResponse)
    
    # Iterate through the response to consume generator
    events = []
    async for event in resp.body_iterator:
        events.append(event)
        
    assert len(events) == 3
    # event 1
    assert events[0]["event"] == "message"
    assert "hello " in events[0]["data"]
    # event 2
    assert events[1]["event"] == "message"
    assert "world" in events[1]["data"]
    # event 3
    assert events[2]["data"] == "[DONE]"
    
    # Assert fallback recorded success
    fallback.record_result.assert_called_once_with("b1", success=True)

@pytest.mark.asyncio
async def test_sse_streamer_disconnect():
    chunk1 = ChatChunk(id="1", model="m1", backend="b1", choices=[])
    
    async def mock_iterator():
        yield chunk1
        
    req = DummyRequest(disconnected=True) # Will disconnect immediately
    
    resp = await create_sse_response(request=req, chunk_iterator=mock_iterator(), model="m1", backend_name="b1", fallback_chain=None)
    
    events = []
    async for event in resp.body_iterator:
        events.append(event)
        
    # Should exit before yielding any events because is_disconnected is true
    assert len(events) == 0

@pytest.mark.asyncio
async def test_sse_streamer_exception():
    async def mock_iterator():
        raise RuntimeError("backend failed")
        yield None
        
    req = DummyRequest()
    fallback = MagicMock()
    
    resp = await create_sse_response(request=req, chunk_iterator=mock_iterator(), model="m1", backend_name="b1", fallback_chain=fallback)
    
    events = []
    async for event in resp.body_iterator:
        events.append(event)
        
    # Should yield error event
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "stream_error" in events[0]["data"]
    
    fallback.record_result.assert_called_once_with("b1", success=False)

@pytest.mark.asyncio
async def test_sse_streamer_generator_exit():
    # GeneratorExit happens when client closes connection abruptly
    async def mock_iterator():
        raise GeneratorExit()
        yield None
        
    req = DummyRequest()
    resp = await create_sse_response(request=req, chunk_iterator=mock_iterator(), model="m1", backend_name="b1", fallback_chain=None)
    
    events = []
    # Using try/except to simulate the runtime behavior if needed, 
    # but GeneratorExit is normally caught cleanly in sse.py
    async for event in resp.body_iterator:
        events.append(event)
        
    assert len(events) == 0

@pytest.mark.asyncio
async def test_sse_streamer_chunk_serialization_error():
    class BadChunk:
        choices = []
        def model_dump_json(self, **kwargs):
            raise ValueError("Cant serialize")
            
    async def mock_iterator():
        yield BadChunk()
        
    req = DummyRequest()
    resp = await create_sse_response(request=req, chunk_iterator=mock_iterator(), model="m1", backend_name="b1", fallback_chain=None)
    
    events = []
    async for event in resp.body_iterator:
        events.append(event)
        
    # Yields [DONE] only because it skips the bad chunk
    assert len(events) == 1
    assert events[0]["data"] == "[DONE]"
