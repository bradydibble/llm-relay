"""Token metrics use the industry-standard direction names and count reasoning.

Old labels were direction="prompt"/"completion" and reasoning was never
recorded at all, so every reasoning field downstream read a hard zero.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry

from llm_relay.metrics import RelayMetrics, set_known_routable
from llm_relay.usage_math import UsageCounts


def _rm() -> RelayMetrics:
    set_known_routable(set())
    return RelayMetrics(registry=CollectorRegistry())


def _sample(rm, metric, **labels):
    for m in rm.registry.collect():
        for s in m.samples:
            if s.name == metric and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


def test_tokens_use_input_and_output_directions():
    rm = _rm()
    rm.record_request(
        alias="main", model="glm-5.2", provider="gb200", outcome="success",
        client="claude-code", usage={"prompt_tokens": 700, "completion_tokens": 90},
        response_body=None, duration_s=1.0, fell_back=False, principal="brady",
    )
    assert _sample(rm, "llm_relay_tokens_total", direction="input", model="glm-5.2") == 700
    assert _sample(rm, "llm_relay_tokens_total", direction="output", model="glm-5.2") == 90
    # The old names must be gone, not aliased.
    assert _sample(rm, "llm_relay_tokens_total", direction="prompt", model="glm-5.2") is None
    assert _sample(rm, "llm_relay_tokens_total", direction="completion", model="glm-5.2") is None


def test_reasoning_tokens_are_recorded_separately():
    rm = _rm()
    rm.record_request(
        alias="gb200", model="glm-5.2", provider="gb200", outcome="success",
        client="claude-code", usage=None, response_body=None,
        duration_s=1.0, fell_back=False, principal="brady",
        counts=UsageCounts(input_tokens=500, output_tokens=400,
                           reasoning_tokens=310, usage_source="upstream_final",
                           reasoning_source="upstream_details"),
    )
    assert _sample(rm, "llm_relay_reasoning_tokens_total", model="glm-5.2") == 310
    # Output still carries the full 400 -- reasoning is an of-which subset.
    assert _sample(rm, "llm_relay_tokens_total", direction="output", model="glm-5.2") == 400


def test_usage_source_is_counted_so_data_quality_is_measurable():
    rm = _rm()
    rm.record_request(
        alias="main", model="ornith-35b", provider="llama-01", outcome="success",
        client="pi", usage=None, response_body=None, duration_s=1.0,
        fell_back=False, principal="brady",
        counts=UsageCounts(input_tokens=10, output_tokens=5,
                           usage_source="frame_count", reasoning_source="none"),
    )
    assert _sample(rm, "llm_relay_usage_source_total", source="frame_count") == 1.0


def test_zero_token_request_records_no_token_samples():
    rm = _rm()
    rm.record_request(
        alias="main", model="ornith-35b", provider="llama-01", outcome="no_backend",
        client="pi", usage={}, response_body=None, duration_s=0.1,
        fell_back=False, principal="brady",
    )
    assert _sample(rm, "llm_relay_tokens_total", direction="input", model="ornith-35b") is None
