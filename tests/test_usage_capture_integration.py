"""End-to-end: an aborted stream must still be accounted for, and every
completion must land in the durable store with its provenance."""
from __future__ import annotations

import os

from llm_relay.api.instrumentation import reassemble_sse, request_shape


def _sse(*frames: str) -> bytes:
    return "".join(f"data: {f}\n\n" for f in frames).encode()


def test_reassemble_counts_frames_for_aborted_streams():
    # Three content deltas and no final usage frame: the stream was cut off.
    raw = _sse(
        '{"choices":[{"delta":{"content":"a"}}]}',
        '{"choices":[{"delta":{"content":"b"}}]}',
        '{"choices":[{"delta":{"content":"c"}}]}',
    )
    text, usage = reassemble_sse(raw)
    assert text == "abc"
    assert usage["_frame_count"] == 3
    assert usage.get("_saw_incremental") is False


def test_reassemble_flags_incremental_usage():
    # Usage on a non-final chunk means continuous_usage_stats is active.
    raw = _sse(
        '{"choices":[{"delta":{"content":"a"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}',
        '{"choices":[{"delta":{"content":"b"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
    )
    _text, usage = reassemble_sse(raw)
    assert usage["_saw_incremental"] is True
    assert usage["completion_tokens"] == 2  # last wins


def test_reassemble_separates_reasoning_text():
    raw = _sse(
        '{"choices":[{"delta":{"reasoning_content":"think"}}]}',
        '{"choices":[{"delta":{"content":"answer"}}]}',
    )
    text, usage = reassemble_sse(raw)
    assert text == "answer"
    assert usage["_reasoning_content"] == "think"


def test_request_shape_fingerprints_without_content():
    body = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ],
        "tools": [{"name": "a"}, {"name": "b"}],
        "temperature": 0.4,
        "max_tokens": 2048,
    }
    shape = request_shape(body)
    assert shape["message_count"] == 4
    assert shape["tool_count"] == 2
    assert shape["temperature"] == 0.4
    assert shape["max_tokens"] == 2048
    assert len(shape["system_hash"]) == 32
    assert len(shape["prefix_hash"]) == 32
    # No content may leak into the fingerprint.
    assert "hello" not in str(shape)


def test_prefix_hash_is_stable_across_a_growing_conversation():
    # The cache-analysis property: turn N+1's prefix is turn N's full history,
    # so a resent conversation is recognisable.
    first = {"messages": [{"role": "user", "content": "q1"}]}
    second = {"messages": [{"role": "user", "content": "q1"},
                           {"role": "assistant", "content": "a1"}]}
    assert request_shape(first)["prefix_hash"] != request_shape(second)["prefix_hash"]
    again = {"messages": [{"role": "user", "content": "q1"},
                          {"role": "assistant", "content": "a1"}]}
    assert request_shape(second)["prefix_hash"] == request_shape(again)["prefix_hash"]


def test_emit_writes_a_row_to_the_store(tmp_path):
    db = str(tmp_path / "usage.db")
    os.environ["LLM_RELAY_USAGE_DB"] = db
    try:
        from llm_relay import usage_store
        from llm_relay.api.instrumentation import emit_chat_completion

        usage_store.reset_store_for_tests()
        emit_chat_completion(
            request_body={"messages": [{"role": "user", "content": "hi"}]},
            response_body={"usage": {"prompt_tokens": 400, "completion_tokens": 25}},
            response_text="ok", usage=None, model_resolved="glm-5.2",
            provider_name="gb200", user_agent="pytest",
            start_ns=1_000_000_000, end_ns=2_000_000_000,
            status_code=200, streamed=False, outcome="success",
            client="claude-code", principal="brady",
        )
        store = usage_store.get_store()
        store.flush()
        conn = usage_store.open_db(db)
        row = conn.execute(
            "SELECT principal, model, input_tokens, output_tokens, usage_source "
            "FROM requests"
        ).fetchone()
        assert row == ("brady", "glm-5.2", 400, 25, "upstream_final")
    finally:
        os.environ.pop("LLM_RELAY_USAGE_DB", None)
        from llm_relay import usage_store as us

        us.reset_store_for_tests()


def test_emit_is_a_noop_when_store_is_unconfigured():
    os.environ.pop("LLM_RELAY_USAGE_DB", None)
    from llm_relay import usage_store
    from llm_relay.api.instrumentation import emit_chat_completion

    usage_store.reset_store_for_tests()
    # Must not raise even though there is nowhere to write.
    emit_chat_completion(
        request_body={"messages": []}, response_body=None, response_text="",
        usage={}, model_resolved="m", provider_name="p", user_agent="pytest",
        start_ns=1, end_ns=2, status_code=200, streamed=False,
        outcome="success", client="c", principal="p",
    )
