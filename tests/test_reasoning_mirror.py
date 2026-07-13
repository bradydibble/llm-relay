"""reasoning -> reasoning_content mirroring.

Some vLLM builds (ornith-397b) emit chain-of-thought in a nonstandard
``reasoning`` field; pi/zed/OpenAI SDKs render ``reasoning_content``. The relay
dual-emits (keeps both) so a client keying off either name sees the thinking.
These cover the two pure transforms — the non-streaming dict path and the
per-frame SSE path — including the passthrough/no-op guarantees that keep the
fast non-reasoning fleet byte-identical.
"""
from __future__ import annotations

import json

from llm_relay.api.app import _mirror_reasoning, _mirror_reasoning_sse_frame


# --- _mirror_reasoning: non-streaming message shape --------------------------

def test_mirror_adds_reasoning_content_on_message():
    payload = {"choices": [{"message": {"role": "assistant", "content": "pong",
                                        "reasoning": "thinking..."}}]}
    assert _mirror_reasoning(payload) is True
    msg = payload["choices"][0]["message"]
    assert msg["reasoning_content"] == "thinking..."
    assert msg["reasoning"] == "thinking..."  # dual-emit: original kept


def test_mirror_preserves_existing_reasoning_content():
    payload = {"choices": [{"message": {"reasoning": "a", "reasoning_content": "b"}}]}
    assert _mirror_reasoning(payload) is False  # do not clobber
    assert payload["choices"][0]["message"]["reasoning_content"] == "b"


def test_mirror_noop_without_reasoning():
    payload = {"choices": [{"message": {"content": "hi"}}]}
    assert _mirror_reasoning(payload) is False
    assert "reasoning_content" not in payload["choices"][0]["message"]


def test_mirror_tolerates_junk_shapes():
    for junk in ({}, {"choices": None}, {"choices": ["x", 1]}, {"choices": [{}]}):
        assert _mirror_reasoning(junk) is False  # never raises


# --- _mirror_reasoning_sse_frame: streaming delta shape ----------------------

def test_sse_frame_mirrors_delta_reasoning():
    obj = {"choices": [{"delta": {"reasoning": "step"}}]}
    frame = "data: " + json.dumps(obj)
    out = _mirror_reasoning_sse_frame(frame)
    line = out.split("data:", 1)[1].strip()
    parsed = json.loads(line)
    assert parsed["choices"][0]["delta"]["reasoning_content"] == "step"
    assert parsed["choices"][0]["delta"]["reasoning"] == "step"


def test_sse_frame_passthrough_done_and_comments():
    # [DONE] sentinel and keepalive comment frames are returned byte-identical.
    assert _mirror_reasoning_sse_frame("data: [DONE]") == "data: [DONE]"
    assert _mirror_reasoning_sse_frame(": ka") == ": ka"


def test_sse_frame_no_reasoning_is_identical_object():
    # Fast-path: a frame with no "reasoning" substring is never parsed/re-serialized.
    frame = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    assert _mirror_reasoning_sse_frame(frame) == frame


def test_sse_frame_non_json_data_line_untouched():
    # "reasoning" present (bypasses the fast-path) but the payload is not JSON:
    # the parse fails and the frame is returned unchanged, never raising.
    frame = "data: reasoning-but-not-json"
    assert _mirror_reasoning_sse_frame(frame) == frame
