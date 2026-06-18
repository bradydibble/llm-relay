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
    set_known_clients,
    set_ua_client_patterns,
    configure_clients_from_env,
    did_fall_back,
    resolve_client,
    client_from_user_agent,
    reset_dynamic_clients,
)


def _rm() -> RelayMetrics:
    # Isolated registry per test; reset alias bounding so unit tests pass through.
    set_known_routable(set())
    return RelayMetrics(registry=CollectorRegistry())


def test_client_from_user_agent_matches_configured_pattern():
    # UA->label patterns are deployment-configured (no agent names ship in the
    # repo). A configured distinctive substring maps to its label.
    set_ua_client_patterns((("agent-x-cli", "agent-x"),))
    try:
        assert client_from_user_agent("agent-x-cli") == "agent-x"
        assert client_from_user_agent("Mozilla/5.0 agent-x-cli extra") == "agent-x"
    finally:
        set_ua_client_patterns(())


def test_client_from_user_agent_returns_none_when_no_pattern_matches():
    # A generic SDK UA matches no configured pattern -> not guessed at (such
    # callers self-identify via the explicit header instead).
    set_ua_client_patterns((("agent-x-cli", "agent-x"),))
    try:
        assert client_from_user_agent("OpenAI/Python 1.30.1") is None
        assert client_from_user_agent("") is None
        assert client_from_user_agent(None) is None
    finally:
        set_ua_client_patterns(())


def test_client_from_user_agent_empty_patterns_is_repo_default():
    # The committed default has no UA patterns: nothing is attributed by UA.
    set_ua_client_patterns(())
    assert client_from_user_agent("agent-x-cli") is None


def test_resolve_client_prefers_explicit_header_over_user_agent():
    set_known_clients({"claude-code", "agent-a"})
    set_ua_client_patterns((("agent-b-cli", "agent-b"),))
    try:
        assert resolve_client("agent-a", "agent-b-cli") == "agent-a"
    finally:
        set_known_clients({"claude-code"})
        set_ua_client_patterns(())


def test_resolve_client_falls_back_to_user_agent_when_header_absent():
    set_ua_client_patterns((("agent-b-cli", "agent-b"),))
    try:
        assert resolve_client(None, "agent-b-cli/0.1") == "agent-b"
        assert resolve_client("", "agent-b-cli") == "agent-b"
    finally:
        set_ua_client_patterns(())


def test_resolve_client_unknown_when_neither_identifies():
    assert resolve_client(None, "OpenAI/Python 1.30.1") == "unknown"
    assert resolve_client(None, None) == "unknown"


def test_resolve_client_honors_self_declared_header_over_ua_sniffing():
    # An explicit header is self-identification: honored as the (sanitized) value
    # the agent declares, never overridden by User-Agent sniffing. No allowlist —
    # a novel name is recorded as itself, not collapsed to "other".
    reset_dynamic_clients()
    set_ua_client_patterns((("agent-b-cli", "agent-b"),))
    try:
        assert resolve_client("some-new-agent", "agent-b-cli") == "some-new-agent"
    finally:
        set_ua_client_patterns(())


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


def test_records_ttft_observation_when_provided():
    # Streaming requests pass a measured time-to-first-token; it lands in the
    # llm_relay_ttft_seconds histogram, labelled by provider/model.
    rm = _rm()
    rm.record_request(
        alias="a", model="m", provider="prov-a", outcome="success",
        client="unknown", usage=None, response_body=None, duration_s=2.0, fell_back=False,
        ttft_s=0.25,
    )
    count = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_count", {"provider": "prov-a", "model": "m"})
    assert count == 1.0


def test_ttft_not_observed_when_none():
    # record_request only observes the ttft it is handed: ttft_s=None -> no
    # observation, so the series is never created. (emit_chat_completion derives
    # a non-streamed ttft from llama.cpp timings upstream of this call.)
    rm = _rm()
    rm.record_request(
        alias="a", model="m", provider="prov-a", outcome="success",
        client="unknown", usage=None, response_body=None, duration_s=2.0, fell_back=False,
        ttft_s=None,
    )
    count = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_count", {"provider": "prov-a", "model": "m"})
    assert count is None


def test_emit_chat_completion_threads_ttft_ns_to_histogram(monkeypatch):
    """emit_chat_completion converts ttft_ns -> seconds and records the histogram."""
    from llm_relay.api.instrumentation import emit_chat_completion
    from llm_relay import metrics as metrics_mod

    rm = RelayMetrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: rm)

    emit_chat_completion(
        request_body={"model": "main"}, response_body=None, response_text="hi", usage=None,
        model_resolved="m", provider_name="prov-a",
        user_agent="ua", start_ns=0, end_ns=1_000_000_000,
        status_code=200, streamed=True, outcome="success",
        ttft_ns=250_000_000,
    )

    count = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_count", {"provider": "prov-a", "model": "m"})
    s = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_sum", {"provider": "prov-a", "model": "m"})
    assert count == 1.0
    assert abs(s - 0.25) < 1e-9


def test_emit_chat_completion_derives_ttft_from_timings_when_not_provided(monkeypatch):
    """A non-streamed llama.cpp response carries server-side prefill time in
    timings.prompt_ms. When the caller doesn't measure an end-to-end ttft_ns (the
    non-streaming path never does), emit_chat_completion derives TTFT from
    prompt_ms so the aggregate histogram isn't streaming-only."""
    from llm_relay.api.instrumentation import emit_chat_completion
    from llm_relay import metrics as metrics_mod

    rm = RelayMetrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: rm)

    emit_chat_completion(
        request_body={"model": "main"},
        response_body={"choices": [], "timings": {"prompt_ms": 1500.0}},
        response_text=None, usage=None,
        model_resolved="m", provider_name="prov-a",
        user_agent="ua", start_ns=0, end_ns=2_000_000_000,
        status_code=200, streamed=False, outcome="success",
        # ttft_ns omitted -> must be derived from timings.prompt_ms
    )

    count = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_count", {"provider": "prov-a", "model": "m"})
    s = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_sum", {"provider": "prov-a", "model": "m"})
    assert count == 1.0
    assert abs(s - 1.5) < 1e-9  # 1500 ms -> 1.5 s


def test_explicit_ttft_ns_wins_over_timings(monkeypatch):
    """The streaming path passes a true end-to-end ttft_ns; it must take
    precedence over any timings.prompt_ms in the body — the measured value is
    better than the prefill proxy, and we never double-count."""
    from llm_relay.api.instrumentation import emit_chat_completion
    from llm_relay import metrics as metrics_mod

    rm = RelayMetrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: rm)

    emit_chat_completion(
        request_body={"model": "main"},
        response_body={"choices": [], "timings": {"prompt_ms": 9000.0}},
        response_text=None, usage=None,
        model_resolved="m", provider_name="prov-a",
        user_agent="ua", start_ns=0, end_ns=2_000_000_000,
        status_code=200, streamed=True, outcome="success",
        ttft_ns=250_000_000,  # explicit end-to-end measurement wins
    )

    s = rm.registry.get_sample_value(
        "llm_relay_ttft_seconds_sum", {"provider": "prov-a", "model": "m"})
    assert abs(s - 0.25) < 1e-9  # 0.25 s, NOT 9.0 s from prompt_ms


def test_fallback_increments_fallbacks_total_only_when_fell_back():
    rm = _rm()
    rm.record_request(
        alias="balanced", model="qwen3.5-9b", provider="prov-a", outcome="success",
        client="claude-code", usage=None, response_body=None, duration_s=1.0, fell_back=True,
    )
    rm.record_request(
        alias="balanced", model="qwen3.5-35b", provider="prov-a", outcome="success",
        client="claude-code", usage=None, response_body=None, duration_s=1.0, fell_back=False,
    )
    fb = rm.registry.get_sample_value(
        "llm_relay_fallbacks_total",
        {"alias": "balanced", "model": "qwen3.5-9b", "client": "claude-code"},
    )
    none_fb = rm.registry.get_sample_value(
        "llm_relay_fallbacks_total",
        {"alias": "balanced", "model": "qwen3.5-35b", "client": "claude-code"},
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


def test_normalize_client_honors_sanitized_self_declared_values():
    reset_dynamic_clients()
    assert normalize_client("claude-code") == "claude-code"
    assert normalize_client("Claude-Code") == "claude-code"  # case-insensitive
    assert normalize_client(None) == "unknown"
    assert normalize_client("") == "unknown"
    assert normalize_client("unknown") == "unknown"  # reserved sentinel
    # A novel self-declared name is honored as itself (no allowlist), not "other".
    assert normalize_client("some-random-script") == "some-random-script"


def test_normalize_client_sanitizes_to_a_safe_label_charset():
    reset_dynamic_clients()
    # Spaces / punctuation collapse to '-', edges trimmed, lower-cased.
    assert normalize_client("My Agent!! v2") == "my-agent-v2"
    assert normalize_client("  weird@@name  ") == "weird-name"
    # All-punctuation sanitizes to empty -> unknown, never a bare "-" label.
    assert normalize_client("@@@") == "unknown"
    # Over-long names are truncated so a single value can't blow up the label.
    assert len(normalize_client("x" * 200)) <= 32


def test_normalize_client_caps_distinct_values_to_protect_cardinality(monkeypatch):
    # Cardinality is bounded WITHOUT a name allowlist: novel labels are honored
    # only up to the cap; beyond it, further new names bucket to "other". An
    # already-seen value and an always-known label stay honored past the cap.
    import llm_relay.metrics as metrics_mod
    reset_dynamic_clients()
    monkeypatch.setattr(metrics_mod, "_MAX_DYNAMIC_CLIENTS", 2)
    assert normalize_client("agent-1") == "agent-1"
    assert normalize_client("agent-2") == "agent-2"
    assert normalize_client("agent-3") == "other"     # cap reached
    assert normalize_client("agent-1") == "agent-1"   # already seen -> still honored
    assert normalize_client("claude-code") == "claude-code"  # known -> cap-exempt
    reset_dynamic_clients()


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


# ---------------------------------------------------------------------------
# Deployment-specific client attribution loaded from the environment
# (keeps real agent names out of the repo; see configure_clients_from_env)
# ---------------------------------------------------------------------------

def test_configure_clients_from_env_loads_known_clients_and_ua_patterns(monkeypatch):
    monkeypatch.setenv("LLM_RELAY_KNOWN_CLIENTS", "agent-a,agent-b")
    monkeypatch.setenv("LLM_RELAY_CLIENT_UA_PATTERNS", "agent-a-cli=agent-a")
    try:
        configure_clients_from_env()
        # claude-code remains a known default; env adds agent-a / agent-b.
        assert normalize_client("agent-a") == "agent-a"
        assert normalize_client("agent-b") == "agent-b"
        assert normalize_client("claude-code") == "claude-code"
        assert client_from_user_agent("agent-a-cli/1.0") == "agent-a"
    finally:
        set_known_clients({"claude-code"})
        set_ua_client_patterns(())


def test_configure_clients_from_env_no_env_keeps_generic_defaults(monkeypatch):
    monkeypatch.delenv("LLM_RELAY_KNOWN_CLIENTS", raising=False)
    monkeypatch.delenv("LLM_RELAY_CLIENT_UA_PATTERNS", raising=False)
    set_known_clients({"claude-code"})
    set_ua_client_patterns(())
    reset_dynamic_clients()
    configure_clients_from_env()
    # The committed default: claude-code is the only always-known label and there
    # is no UA attribution. A name like agent-a isn't pre-registered, but self-
    # identification is still honored generically (bounded by the cardinality cap)
    # rather than gated by an allowlist of deployment names.
    assert normalize_client("claude-code") == "claude-code"
    assert normalize_client("agent-a") == "agent-a"
    assert client_from_user_agent("agent-a-cli") is None
