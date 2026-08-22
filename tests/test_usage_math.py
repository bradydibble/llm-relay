"""Tests for llm_relay.usage_math — the single place that decides how many
tokens a request used and how confident we are in that number."""
from __future__ import annotations

from llm_relay.usage_math import (
    REASON_DETAILS,
    REASON_NONE,
    REASON_SPLIT,
    SOURCE_ESTIMATE,
    SOURCE_FINAL,
    SOURCE_FRAMES,
    SOURCE_INCREMENTAL,
    SOURCE_NONE,
    estimate_tokens,
    resolve_usage,
)


def test_exact_usage_from_response_body_is_final():
    # Non-streaming: usage lives in the response body and is authoritative.
    r = resolve_usage(
        usage=None,
        response_body={"usage": {"prompt_tokens": 1200, "completion_tokens": 340}},
        streamed=False,
    )
    assert r.input_tokens == 1200
    assert r.output_tokens == 340
    assert r.usage_source == SOURCE_FINAL


def test_incremental_usage_is_flagged_as_incremental():
    # vLLM continuous_usage_stats: usage rode on every chunk, so even a stream
    # that died mid-flight has exact numbers.
    r = resolve_usage(
        usage={"prompt_tokens": 900, "completion_tokens": 55},
        response_body=None,
        streamed=True,
        saw_incremental=True,
    )
    assert r.input_tokens == 900
    assert r.output_tokens == 55
    assert r.usage_source == SOURCE_INCREMENTAL


def test_aborted_stream_without_usage_falls_back_to_frame_count():
    # THE BUG: an interrupted stream used to record zero. llama.cpp emits one
    # SSE delta frame per token, so frames are a good output count.
    r = resolve_usage(
        usage={},
        response_body=None,
        streamed=True,
        frame_count=128,
        content_text="x" * 400,
    )
    assert r.output_tokens == 128
    assert r.usage_source == SOURCE_FRAMES


def test_aborted_stream_keeps_prompt_tokens_when_timings_seen():
    # An aborted stream that saw llama.cpp timings must not throw away the
    # input tokens it already consumed.
    r = resolve_usage(
        usage={"prompt_tokens": 74000},
        response_body=None,
        streamed=True,
        frame_count=12,
    )
    assert r.input_tokens == 74000
    assert r.output_tokens == 12
    assert r.usage_source == SOURCE_FRAMES


def test_no_frames_and_no_usage_falls_back_to_tokenizer_estimate():
    r = resolve_usage(
        usage={},
        response_body=None,
        streamed=True,
        frame_count=0,
        content_text="hello world this is some generated text",
    )
    assert r.output_tokens == estimate_tokens("hello world this is some generated text")
    assert r.output_tokens > 0
    assert r.usage_source == SOURCE_ESTIMATE


def test_failed_request_with_no_tokens_reports_source_none():
    r = resolve_usage(usage={}, response_body=None, streamed=False)
    assert r.input_tokens == 0
    assert r.output_tokens == 0
    assert r.usage_source == SOURCE_NONE
    assert r.reasoning_source == REASON_NONE


def test_reasoning_from_upstream_details_is_exact():
    # vLLM reports the split directly; trust it.
    r = resolve_usage(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 500,
            "completion_tokens_details": {"reasoning_tokens": 380},
        },
        response_body=None,
        streamed=True,
        saw_incremental=True,
    )
    assert r.output_tokens == 500
    assert r.reasoning_tokens == 380
    assert r.reasoning_source == REASON_DETAILS


def test_reasoning_split_proportionally_when_details_absent():
    # 750 reasoning chars vs 250 content chars = 75% of an EXACT 400 output.
    r = resolve_usage(
        usage={"prompt_tokens": 10, "completion_tokens": 400},
        response_body=None,
        streamed=True,
        content_text="c" * 250,
        reasoning_text="r" * 750,
    )
    assert r.output_tokens == 400          # total stays exact
    assert r.reasoning_tokens == 300       # only the split is approximate
    assert r.reasoning_source == REASON_SPLIT


def test_reasoning_never_exceeds_output():
    # Guard against a bad upstream number corrupting the invariant.
    r = resolve_usage(
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 900},
        },
        response_body=None,
        streamed=False,
    )
    assert r.reasoning_tokens <= r.output_tokens


def test_no_reasoning_text_means_no_reasoning_tokens():
    r = resolve_usage(
        usage={"prompt_tokens": 10, "completion_tokens": 400},
        response_body=None,
        streamed=True,
        content_text="c" * 250,
        reasoning_text="",
    )
    assert r.reasoning_tokens == 0
    assert r.reasoning_source == REASON_NONE


def test_cache_read_tokens_come_from_llamacpp_timings():
    r = resolve_usage(
        usage={"prompt_tokens": 500, "completion_tokens": 20},
        response_body={"timings": {"cache_n": 480}},
        streamed=False,
    )
    assert r.cache_read_tokens == 480


def test_negative_and_garbage_values_are_coerced_not_raised():
    r = resolve_usage(
        usage={"prompt_tokens": -5, "completion_tokens": "abc"},
        response_body=None,
        streamed=False,
    )
    assert r.input_tokens == 0
    assert r.output_tokens == 0


def test_estimate_tokens_is_monotonic_and_nonzero_for_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a short line") > 0
    assert estimate_tokens("a" * 1000) > estimate_tokens("a" * 10)


def test_cache_read_from_openai_standard_field():
    # The OpenAI-standard location. Reading only llama.cpp's timings.cache_n
    # measured one backend family and silently reported 0 for the rest.
    r = resolve_usage(
        usage={"prompt_tokens": 2025, "completion_tokens": 4,
               "prompt_tokens_details": {"cached_tokens": 2021}},
        response_body=None,
        streamed=False,
    )
    assert r.cache_read_tokens == 2021


def test_cache_read_falls_back_to_llamacpp_timings():
    r = resolve_usage(
        usage={"prompt_tokens": 500, "completion_tokens": 20},
        response_body={"timings": {"cache_n": 480}},
        streamed=False,
    )
    assert r.cache_read_tokens == 480


def test_cache_read_prefers_the_standard_field_over_timings():
    r = resolve_usage(
        usage={"prompt_tokens": 500, "completion_tokens": 20,
               "prompt_tokens_details": {"cached_tokens": 300}},
        response_body={"timings": {"cache_n": 480}},
        streamed=False,
    )
    assert r.cache_read_tokens == 300


def test_cache_read_never_exceeds_the_prompt_it_reused():
    r = resolve_usage(
        usage={"prompt_tokens": 100, "completion_tokens": 5,
               "prompt_tokens_details": {"cached_tokens": 99999}},
        response_body=None,
        streamed=False,
    )
    assert r.cache_read_tokens == 100


def test_absent_cache_reporting_is_zero_not_an_error():
    # vLLM reports neither field; zero here means "not reported", which is NOT
    # evidence that no reuse occurred.
    r = resolve_usage(
        usage={"prompt_tokens": 2027, "completion_tokens": 4},
        response_body=None,
        streamed=False,
    )
    assert r.cache_read_tokens == 0
