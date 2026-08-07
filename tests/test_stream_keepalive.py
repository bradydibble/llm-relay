"""Streaming prefill keepalive (fixes large-context ornith TTFT connection death)."""
import asyncio
import pytest
from llm_relay.api.app import _sse_stream_keepalive


async def _iter(delay, chunks):
    if delay:
        await asyncio.sleep(delay)
    for c in chunks:
        yield c


async def _drain(gen):
    ka, data = [], []
    async for payload, is_ka in gen:
        (ka if is_ka else data).append(payload)
    return ka, data


async def test_keepalive_emitted_during_prefill_silence():
    ka, data = await _drain(
        _sse_stream_keepalive(_iter(0.35, [b"data: a\n\n", b"data: b\n\n"]),
                              "text/event-stream", 0.1))
    assert len(ka) >= 2 and all(p == b": ka\n\n" for p in ka)
    assert data == [b"data: a\n\n", b"data: b\n\n"]


async def test_no_keepalive_when_data_flows_fast():
    ka, data = await _drain(
        _sse_stream_keepalive(_iter(0.0, [b"x", b"y", b"z"]), "text/event-stream", 0.1))
    assert ka == []
    assert data == [b"x", b"y", b"z"]


async def test_non_sse_media_type_passes_through_untouched():
    ka, data = await _drain(
        _sse_stream_keepalive(_iter(0.25, [b"{}"]), "application/json", 0.05))
    assert ka == []
    assert data == [b"{}"]


async def test_pending_task_cancelled_on_early_close():
    gen = _sse_stream_keepalive(_iter(5.0, [b"late"]), "text/event-stream", 0.05)
    first = await gen.__anext__()
    assert first == (b": ka\n\n", True)
    await gen.aclose()  # must not hang or leak the pending __anext__ task
