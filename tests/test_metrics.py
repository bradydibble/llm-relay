"""Tests for the Prometheus metrics layer (llm_relay.metrics).

The metrics layer is deliberately decoupled from the OTLP span path: it must
record request/token/fallback counters and expose backend gauges regardless of
whether telemetry (Phoenix) is enabled or reachable.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry

from llm_relay.metrics import (
    RelayMetrics,
    DiscoveryCollector,
    normalize_client,
    normalize_alias,
    set_known_routable,
    did_fall_back,
)


def _rm() -> RelayMetrics:
    # Isolated registry per test; reset alias bounding so unit tests pass through.
    set_known_routable(set())
    return RelayMetrics(registry=CollectorRegistry())


def test_normalize_alias_passes_through_when_no_known_set_registered():
    set_known_routable(set())
    assert normalize_alias("anything-goes") == "anything-goes"
    assert normalize_alias(None) == "none"


def test_normalize_alias_bounds_unknown_to_other_when_known_set_registered():
    set_known_routable({"balanced", "qwen3.5-35b"})
    assert normalize_alias("balanced") == "balanced"
    assert normalize_alias("totally-made-up") == "other"
    set_known_routable(set())  # reset so we don't leak into other tests


def test_did_fall_back_true_when_selected_is_not_the_preferred_candidate():
    assert did_fall_back("model-b", ["model-a", "model-b"]) is True


def test_did_fall_back_false_when_selected_is_the_preferred_candidate():
    assert did_fall_back("model-a", ["model-a", "model-b"]) is False


def test_did_fall_back_false_when_ranked_empty_or_selected_missing():
    assert did_fall_back("model-a", []) is False
    assert did_fall_back(None, ["model-a"]) is False


def test_records_one_request_against_provider_model_alias_outcome_client():
    rm = _rm()
    rm.record_request(
        alias="balanced", model="qwen3.5-35b", provider="prov-a",
        outcome="success", client="claude-code",
        usage=None, response_body=None, duration_s=1.5, fell_back=False,
    )
    v = rm.registry.get_sample_value(
        "llm_relay_requests_total",
        {"provider": "prov-a", "model": "qwen3.5-35b",
         "alias": "balanced", "outcome": "success", "client": "claude-code"},
    )
    assert v == 1.0


def test_counts_prompt_and_completion_tokens_from_streaming_usage():
    rm = _rm()
    rm.record_request(
        alias="balanced", model="m", provider="prov-a", outcome="success",
        client="unknown", usage={"prompt_tokens": 100, "completion_tokens": 40},
        response_body=None, duration_s=1.0, fell_back=False,
    )
    prompt = rm.registry.get_sample_value(
        "llm_relay_tokens_total",
        {"provider": "prov-a", "model": "m", "direction": "prompt", "client": "unknown"},
    )
    completion = rm.registry.get_sample_value(
        "llm_relay_tokens_total",
        {"provider": "prov-a", "model": "m", "direction": "completion", "client": "unknown"},
    )
    assert prompt == 100.0
    assert completion == 40.0


def test_non_streaming_tokens_extracted_from_response_body_usage():
    # Non-streaming success passes usage=None and the full response_body, whose
    # ["usage"] holds the counts. Naive wiring records zero here.
    rm = _rm()
    rm.record_request(
        alias="balanced", model="m", provider="prov-a", outcome="success",
        client="unknown", usage=None,
        response_body={"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 12}},
        duration_s=1.0, fell_back=False,
    )
    completion = rm.registry.get_sample_value(
        "llm_relay_tokens_total",
        {"provider": "prov-a", "model": "m", "direction": "completion", "client": "unknown"},
    )
    assert completion == 12.0


def test_records_request_duration_observation():
    rm = _rm()
    rm.record_request(
        alias="a", model="m", provider="prov-a", outcome="success",
        client="unknown", usage=None, response_body=None, duration_s=2.0, fell_back=False,
    )
    count = rm.registry.get_sample_value(
        "llm_relay_request_duration_seconds_count",
        {"provider": "prov-a", "model": "m"},
    )
    assert count == 1.0


def test_fallback_increments_fallbacks_total_only_when_fell_back():
    rm = _rm()
    rm.record_request(
        alias="balanced", model="qwen3.5-9b", provider="prov-a", outcome="success",
        client="agent-b", usage=None, response_body=None, duration_s=1.0, fell_back=True,
    )
    rm.record_request(
        alias="balanced", model="qwen3.5-35b", provider="prov-a", outcome="success",
        client="agent-b", usage=None, response_body=None, duration_s=1.0, fell_back=False,
    )
    fb = rm.registry.get_sample_value(
        "llm_relay_fallbacks_total",
        {"alias": "balanced", "model": "qwen3.5-9b", "client": "agent-b"},
    )
    none_fb = rm.registry.get_sample_value(
        "llm_relay_fallbacks_total",
        {"alias": "balanced", "model": "qwen3.5-35b", "client": "agent-b"},
    )
    assert fb == 1.0
    assert none_fb is None  # the non-fallback request created no fallback series


def test_none_provider_and_model_coerced_to_label_safe_string():
    # Error paths pass model/provider = None; labels must not be None.
    rm = _rm()
    rm.record_request(
        alias="balanced", model=None, provider=None, outcome="no_candidate",
        client="unknown", usage=None, response_body=None, duration_s=0.1, fell_back=False,
    )
    v = rm.registry.get_sample_value(
        "llm_relay_requests_total",
        {"provider": "none", "model": "none", "alias": "balanced",
         "outcome": "no_candidate", "client": "unknown"},
    )
    assert v == 1.0


def test_normalize_client_buckets_to_known_or_other_or_unknown():
    assert normalize_client("claude-code") == "claude-code"
    assert normalize_client("Claude-Code") == "claude-code"  # case-insensitive
    assert normalize_client(None) == "unknown"
    assert normalize_client("") == "unknown"
    assert normalize_client("some-random-script") == "other"  # cardinality guard


class _FakeState:
    def __init__(self, provider, status, models, circuit_open):
        self.provider = provider
        self.status = status
        self.models = models
        self.circuit_open = circuit_open


class _FakeStatus:
    def __init__(self, value):
        self.value = value


class _FakeClient:
    def __init__(self, provider, status_value, models, circuit_open, inflight, cap):
        self.state = _FakeState(provider, _FakeStatus(status_value), models, circuit_open)
        self.inflight_used = inflight
        self.max_concurrent = cap


class _FakeDiscovery:
    def __init__(self, clients):
        self.clients = clients


def test_discovery_collector_exposes_backend_gauges_at_scrape():
    disc = _FakeDiscovery({
        "prov-a:8080": _FakeClient("prov-a", "healthy", ["qwen3.5-9b"], False, 2, 3),
        "prov-b:8000": _FakeClient("prov-b", "unavailable", [], True, 0, 2),
    })
    collector = DiscoveryCollector(disc)
    samples = {}
    for metric in collector.collect():
        for s in metric.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value

    def get(name, **labels):
        return samples.get((name, tuple(sorted(labels.items()))))

    # prov-a healthy → up=1, inflight=2, cap=3, circuit closed=0
    assert get("llm_relay_backend_up", backend="prov-a:8080", provider="prov-a") == 1.0
    assert get("llm_relay_inflight_requests", backend="prov-a:8080", provider="prov-a") == 2.0
    assert get("llm_relay_backend_max_concurrent", backend="prov-a:8080", provider="prov-a") == 3.0
    assert get("llm_relay_circuit_breaker_state", backend="prov-a:8080", provider="prov-a") == 0.0
    # prov-b unavailable + circuit open
    assert get("llm_relay_backend_up", backend="prov-b:8000", provider="prov-b") == 0.0
    assert get("llm_relay_circuit_breaker_state", backend="prov-b:8000", provider="prov-b") == 1.0


def test_discovery_collector_exposes_reconcile_and_reset_counters():
    """The leaked-slot containment paths (reconcile, backend-wipe) must surface
    as counters so an operator can see whether they ever fire in production."""
    c = _FakeClient("prov-a", "healthy", ["m"], False, 0, 3)
    c.slot_reconciliations = 2
    c.backend_resets = 1
    collector = DiscoveryCollector(_FakeDiscovery({"prov-a:8080": c}))

    samples = {}
    for metric in collector.collect():
        for s in metric.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value

    def get(name, **labels):
        return samples.get((name, tuple(sorted(labels.items()))))

    assert get("llm_relay_slot_reconciliations_total", backend="prov-a:8080", provider="prov-a") == 2.0
    assert get("llm_relay_backend_resets_total", backend="prov-a:8080", provider="prov-a") == 1.0
