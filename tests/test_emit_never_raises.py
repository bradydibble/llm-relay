"""emit_chat_completion must not raise into the request path.

It is called unguarded from the chat-completions handler, so anything that
escapes here turns a SUCCESSFUL completion into a 500 and throws the answer
away. A backend is free to return junk in fields we only read for telemetry.
"""
from __future__ import annotations

import pytest

from llm_relay.api.instrumentation import emit_chat_completion


def _emit(**over):
    kwargs = dict(
        request_body={"messages": [{"role": "user", "content": "hi"}]},
        response_body={"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        response_text="ok", usage=None, model_resolved="m", provider_name="p",
        user_agent="pytest", start_ns=1_000_000_000, end_ns=2_000_000_000,
        status_code=200, streamed=False, outcome="success",
        client="c", principal="brady",
    )
    kwargs.update(over)
    emit_chat_completion(**kwargs)


@pytest.mark.parametrize("timings", ["not-a-dict", 42, [1, 2], True])
def test_non_dict_timings_does_not_raise(timings):
    # `(body.get("timings") or {})` does NOT protect against a truthy non-dict:
    # "str".get -> AttributeError, straight out of the request path.
    _emit(response_body={"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                         "timings": timings})


@pytest.mark.parametrize("frame_count", ["three", None, [], {"a": 1}, "12x"])
def test_garbage_frame_count_does_not_raise(frame_count):
    # int("three") -> ValueError. _frame_count is set by our own reassembler
    # today, but it is read off an untrusted dict with no isinstance guard.
    _emit(usage={"prompt_tokens": 1, "completion_tokens": 1,
                 "_frame_count": frame_count}, response_body=None, streamed=True)


def test_garbage_reasoning_content_does_not_raise():
    _emit(usage={"prompt_tokens": 1, "completion_tokens": 1,
                 "_reasoning_content": {"nested": "obj"}},
          response_body=None, streamed=True)


def test_non_dict_response_body_does_not_raise():
    _emit(response_body="a plain string body")
