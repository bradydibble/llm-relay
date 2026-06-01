"""Streaming outcome fidelity (F3): classify how a stream actually terminated,
not just its initial HTTP status, so a 200-then-stall stops being logged as
success.

These are pure-function tests. The wiring into _tee_and_emit is integration-
tested separately — a mid-stream client disconnect resists ASGITransport, which
is exactly why the classification logic is extracted into a pure function here.
"""
from __future__ import annotations

import asyncio

import httpx

from llm_relay.api.instrumentation import _classify_stream_outcome, sse_finished


# Priority order under test: error status wins, then client cancellation, then
# a mid-stream error, then clean-finish vs truncated.

def test_classify_clean_finish_is_success():
    assert _classify_stream_outcome(200, None, True) == "success"


def test_classify_ended_without_terminal_marker_is_incomplete():
    # 200 but the stream stopped without [DONE]/finish_reason — the silent hangup
    # that today is mislabeled "success".
    assert _classify_stream_outcome(200, None, False) == "stream_incomplete"


def test_classify_client_cancelled_is_client_disconnect():
    assert _classify_stream_outcome(200, asyncio.CancelledError(), False) == "client_disconnect"


def test_classify_generator_exit_is_client_disconnect():
    assert _classify_stream_outcome(200, GeneratorExit(), False) == "client_disconnect"


def test_classify_midstream_exception_is_stream_error():
    assert _classify_stream_outcome(200, httpx.RemoteProtocolError("peer closed"), False) == "stream_error"
    assert _classify_stream_outcome(200, RuntimeError("boom"), False) == "stream_error"


def test_classify_error_status_wins_over_finish_state():
    # The all-retryable-5xx stream returned by the streaming spill drains cleanly
    # WITHOUT a [DONE]; it must read upstream_error, not stream_incomplete.
    assert _classify_stream_outcome(503, None, False) == "upstream_error"


def test_classify_error_status_wins_over_client_disconnect():
    # If the upstream errored AND the client then bailed, the upstream error is
    # the meaningful outcome — status is checked first.
    assert _classify_stream_outcome(502, asyncio.CancelledError(), False) == "upstream_error"


# --- sse_finished: clean termination via [DONE] sentinel or non-null finish_reason ---

def test_sse_finished_true_on_done_sentinel():
    assert sse_finished(b"data: {}\n\ndata: [DONE]\n\n") is True


def test_sse_finished_true_on_finish_reason():
    raw = b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
    assert sse_finished(raw) is True


def test_sse_finished_false_when_finish_reason_null():
    # Intermediate chunks carry finish_reason=null; that is NOT a clean finish.
    raw = b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    assert sse_finished(raw) is False


def test_sse_finished_false_when_truncated():
    # Deltas but no [DONE] and no finish_reason — the stream was cut off.
    raw = b'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'
    assert sse_finished(raw) is False


def test_sse_finished_false_on_empty():
    assert sse_finished(b"") is False
