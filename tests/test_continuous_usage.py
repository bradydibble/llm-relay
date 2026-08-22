"""vLLM's continuous_usage_stats puts token usage on EVERY streamed chunk, so a
client that disconnects mid-generation still leaves exact counts behind."""
from __future__ import annotations

import os

from llm_relay.routing.router import apply_stream_usage_options, continuous_usage_enabled


def test_include_usage_is_always_requested():
    body = {"model": "m", "stream": True}
    apply_stream_usage_options(body)
    assert body["stream_options"]["include_usage"] is True


def test_continuous_usage_off_by_default():
    os.environ.pop("LLM_RELAY_CONTINUOUS_USAGE", None)
    assert continuous_usage_enabled() is False
    body = {"model": "m", "stream": True}
    apply_stream_usage_options(body)
    assert "continuous_usage_stats" not in body["stream_options"]


def test_continuous_usage_added_when_enabled():
    os.environ["LLM_RELAY_CONTINUOUS_USAGE"] = "1"
    try:
        assert continuous_usage_enabled() is True
        body = {"model": "m", "stream": True}
        apply_stream_usage_options(body)
        assert body["stream_options"]["continuous_usage_stats"] is True
    finally:
        os.environ.pop("LLM_RELAY_CONTINUOUS_USAGE", None)


def test_caller_overrides_are_preserved():
    # A client that explicitly opted out must stay opted out.
    os.environ["LLM_RELAY_CONTINUOUS_USAGE"] = "1"
    try:
        body = {"model": "m", "stream": True,
                "stream_options": {"include_usage": False,
                                   "continuous_usage_stats": False}}
        apply_stream_usage_options(body)
        assert body["stream_options"]["include_usage"] is False
        assert body["stream_options"]["continuous_usage_stats"] is False
    finally:
        os.environ.pop("LLM_RELAY_CONTINUOUS_USAGE", None)
