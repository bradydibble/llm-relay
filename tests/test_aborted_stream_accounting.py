"""A truncated stream must still be accounted for, end to end.

This is the headline defect the usage store exists to fix. Token usage arrives
only in the final SSE chunk, so a stream that ends early — a client that hit
Ctrl-C, a backend that hung up — used to contribute NOTHING: not the output it
had already generated, and not the (often enormous) prompt it had already
consumed. Agent clients interrupt constantly, so this was not an edge case; it
was a systematic undercount.

``tests/test_usage_math.py`` covers the arithmetic and
``tests/test_usage_capture_integration.py`` covers ``reassemble_sse`` in
isolation. Neither proves the fallback actually fires through the real
streaming route, which is what these tests do: drive a genuine request through
``/v1/chat/completions`` with the real ``emit_chat_completion`` in place, and
assert a row lands in the store with a non-zero count and honest provenance.

A true socket-level disconnect resists ASGITransport (see the note in
tests/test_stream_lifecycle.py), so the trigger here is the same one the
outcome classifier uses: a 200 stream that simply ends without a terminal
marker or a usage chunk. That reaches the identical code path — the streaming
``finally`` reassembles whatever arrived and emits from it.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from httpx import ASGITransport

from llm_relay import usage_store
from llm_relay.api.app import create_app
from llm_relay.routing.router import RouteResult


def _make_minimal_config(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {"type": "openai", "base_url": "http://127.0.0.1",
                          "ownership": "ciq_owned", "enabled": True},
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "test-model": {"provider": "local-llm", "class": "unknown",
                           "privacy": "local_only"},
        }
    }))
    return cfg_dir


def _streaming_app(tmp_path, monkeypatch, body_chunks, status=200):
    """App whose streaming route yields ``body_chunks``.

    Unlike the spy harness in test_stream_lifecycle.py, the REAL
    emit_chat_completion runs here — that is the point: we are testing that the
    durable row actually gets written, not just that emit was called.
    """
    app = create_app(config_dir=_make_minimal_config(tmp_path))

    async def _cleanup() -> None:
        return None

    async def _body():
        for c in body_chunks:
            yield c

    async def _rf(request_data, headers=None, stream=False):
        result = RouteResult(
            success=True, selected_model="test-model",
            backend_url="http://127.0.0.1", provider_name="local-llm",
            decision={"ranked": ["test-model"]},
        )
        upstream = httpx.Response(status, headers={"content-type": "text/event-stream"})
        return upstream, _body(), result, _cleanup

    monkeypatch.setattr(app.state.router, "route_and_forward", _rf)
    return app


async def _post_stream(app, messages=None):
    body = {
        "model": "test-model",
        "messages": messages or [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        resp = await http.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        _ = resp.content  # drain so the streaming finally runs
    return resp


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Enable the usage store at a temp path for the duration of one test."""
    db = str(tmp_path / "usage.db")
    monkeypatch.setenv("LLM_RELAY_USAGE_DB", db)
    usage_store.reset_store_for_tests()
    yield db
    usage_store.reset_store_for_tests()


def _rows(db):
    conn = usage_store.open_db(db)
    try:
        return conn.execute(
            "SELECT request_id, model, input_tokens, output_tokens, "
            "reasoning_tokens, usage_source, outcome, streamed FROM requests"
        ).fetchall()
    finally:
        conn.close()


async def test_truncated_stream_is_counted_not_dropped(tmp_path, monkeypatch, store):
    # Three content deltas and then nothing: no [DONE], no usage chunk. Before
    # the fix this recorded zero tokens; now the frames themselves are the count.
    app = _streaming_app(tmp_path, monkeypatch, [
        b'data: {"choices":[{"delta":{"content":"par"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ti"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"al"}}]}\n\n',
    ])
    await _post_stream(app)
    usage_store.get_store().flush()

    rows = _rows(store)
    assert len(rows) == 1, "a truncated stream must still produce a usage row"
    _rid, model, inp, out, reasoning, source, outcome, streamed = rows[0]
    assert model == "test-model"
    assert out == 3, "one SSE delta frame per token"
    assert source == "frame_count", "and it must say the count was inferred"
    assert reasoning == 0
    assert streamed == 1
    # The outcome stays honest about how the stream ended.
    assert outcome == "stream_incomplete"


async def test_clean_stream_with_usage_is_exact(tmp_path, monkeypatch, store):
    app = _streaming_app(tmp_path, monkeypatch, [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1234,"completion_tokens":7}}\n\n',
        b"data: [DONE]\n\n",
    ])
    await _post_stream(app)
    usage_store.get_store().flush()

    rows = _rows(store)
    assert len(rows) == 1
    _rid, _m, inp, out, _r, source, outcome, _s = rows[0]
    assert (inp, out) == (1234, 7)
    # Usage on a choices-empty terminal chunk is the standard include_usage
    # shape, so this is 'final', not 'incremental'.
    assert source == "upstream_final"
    assert outcome == "success"


async def test_incremental_usage_survives_a_truncated_stream(tmp_path, monkeypatch, store):
    # vLLM's continuous_usage_stats puts usage on every chunk. A stream cut off
    # after the second chunk therefore still has EXACT counts, including the
    # prompt tokens already consumed — no estimation at all.
    app = _streaming_app(tmp_path, monkeypatch, [
        b'data: {"choices":[{"delta":{"content":"a"}}],"usage":{"prompt_tokens":90000,"completion_tokens":1}}\n\n',
        b'data: {"choices":[{"delta":{"content":"b"}}],"usage":{"prompt_tokens":90000,"completion_tokens":2}}\n\n',
    ])
    await _post_stream(app)
    usage_store.get_store().flush()

    rows = _rows(store)
    assert len(rows) == 1
    _rid, _m, inp, out, _r, source, _o, _s = rows[0]
    assert inp == 90000, "the prompt was consumed and must be billed"
    assert out == 2, "last incremental usage wins"
    assert source == "upstream_incremental"


async def test_reasoning_is_split_out_of_output_on_a_real_stream(tmp_path, monkeypatch, store):
    # 600 reasoning chars vs 200 content chars, with an exact output total of 80:
    # reasoning should take ~3/4 of it, and must never exceed it.
    app = _streaming_app(tmp_path, monkeypatch, [
        b'data: {"choices":[{"delta":{"reasoning_content":"' + b"r" * 600 + b'"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"' + b"c" * 200 + b'"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":80}}\n\n',
        b"data: [DONE]\n\n",
    ])
    await _post_stream(app)
    usage_store.get_store().flush()

    rows = _rows(store)
    assert len(rows) == 1
    _rid, _m, _i, out, reasoning, _src, _o, _s = rows[0]
    assert out == 80, "the output total stays exact"
    assert reasoning == 60, "reasoning is a proportional slice of that exact total"
    assert reasoning <= out, "reasoning is a subset of output, never a sibling"


async def test_request_fingerprint_is_recorded_without_content(tmp_path, monkeypatch, store):
    app = _streaming_app(tmp_path, monkeypatch, [
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
        b"data: [DONE]\n\n",
    ])
    await _post_stream(app, messages=[
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "a-very-distinctive-secret-phrase"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ])
    usage_store.get_store().flush()

    conn = usage_store.open_db(store)
    try:
        row = conn.execute(
            "SELECT message_count, system_hash, prefix_hash, tool_count FROM requests"
        ).fetchone()
    finally:
        conn.close()
    message_count, system_hash, prefix_hash, tool_count = row
    assert message_count == 4
    assert tool_count == 0
    assert system_hash and prefix_hash
    # The fingerprint is structural: no prompt text may reach the store here.
    assert "a-very-distinctive-secret-phrase" not in str(row)
