"""Span-attribute capture under OpenTelemetry's per-span attribute limit.

OTel caps a span at SpanLimits.max_attributes (default 128) and evicts the
OLDEST attribute when the limit is hit. The relay records one attribute per chat
message, which is unbounded with conversation length, so the critical routing
attributes (model / provider / outcome / span.kind) must be written LAST to
survive. Regression guard: before the reorder, a ~60-message request silently
dropped model_name + provider and rendered in Phoenix as an untyped span.
"""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider, SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from llm_relay.api import instrumentation as ins
from llm_relay import metrics as metrics_mod
from llm_relay.metrics import RelayMetrics


def _capture_span(monkeypatch, *, messages, max_attributes=128, **emit_kwargs):
    """Run emit_chat_completion against an in-memory tracer at a fixed attribute
    limit and return the finished span's attributes as a dict."""
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: RelayMetrics(registry=CollectorRegistry()))

    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=SpanLimits(max_attributes=max_attributes))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(ins, "_INITIALIZED", True)
    monkeypatch.setattr(ins, "_TRACER", provider.get_tracer("test"))

    defaults = dict(
        request_body={"model": "main", "messages": messages},
        response_body=None, response_text="ok",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        model_resolved="qwen3.5-9b", provider_name="local-llm",
        user_agent="ua", start_ns=0, end_ns=1_000_000_000,
        status_code=200, streamed=True, outcome="success",
    )
    defaults.update(emit_kwargs)
    ins.emit_chat_completion(**defaults)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return dict(spans[0].attributes)


def test_long_conversation_keeps_model_provider_and_outcome(monkeypatch):
    """A conversation long enough to blow past the 128-attribute limit must still
    carry model / provider / outcome / span.kind — the whole point of the span."""
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                for i in range(120)]  # 240 per-message attrs -> well over the 128 cap
    attrs = _capture_span(monkeypatch, messages=messages)

    assert attrs.get("llm.model_name") == "qwen3.5-9b"
    assert attrs.get("llm.provider") == "local-llm"
    assert attrs.get("llm.relay.outcome") == "success"
    assert attrs.get("openinference.span.kind") == "LLM"
    # The full prompt is preserved in the value blob even when per-message keys
    # were evicted to make room.
    assert "input.value" in attrs


def test_short_conversation_keeps_everything(monkeypatch):
    """A small request stays under the limit and retains both the routing
    attributes and the per-message breakdown."""
    messages = [{"role": "user", "content": "hello"}]
    attrs = _capture_span(monkeypatch, messages=messages)

    assert attrs.get("llm.model_name") == "qwen3.5-9b"
    assert attrs.get("llm.provider") == "local-llm"
    assert attrs.get("llm.input_messages.0.message.content") == "hello"


def test_rejected_request_has_outcome_but_no_model(monkeypatch):
    """Pre-backend rejections (saturated / network_error) resolve no model, so
    model/provider are legitimately absent — but outcome must still be recorded."""
    messages = [{"role": "user", "content": "x"}]
    attrs = _capture_span(monkeypatch, messages=messages,
                          model_resolved=None, provider_name=None,
                          outcome="saturated", status_code=503,
                          response_text=None, usage=None)

    assert attrs.get("llm.relay.outcome") == "saturated"
    assert "llm.model_name" not in attrs
    assert "llm.provider" not in attrs
