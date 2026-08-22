"""Conformance matrix: every code path applies the SAME safety defaults.

This test was prompted by the 2026-08-22 gap audit: the streaming path in
``route_and_forward`` did NOT apply ``max_tokens`` or ``repetition_penalty``
defaults (non-streaming did), letting the same payload intermittently hang
depending on the ``stream`` flag. The matrix below would have caught that gap
_before_ it shipped by asserting equivalence across every dimension:

    {streaming, non-streaming}
      x {max_tokens absent, set_huge, set_safe, max_completion_tokens only, both mixed}
      x {repetition_penalty absent, explicitly 1.0, explicitly 1.2}

Each cell exercises BOTH paths with the same payload and asserts the forwarded
values are identical. Additional layers cover:
  - ``max_completion_tokens`` (OpenAI's Nov 2024 rename) equivalence to ``max_tokens``
  - ``set_params`` cannot override safety defaults at runtime
  - config-loader warning for safety-sensitive ``set_params``
"""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest
import yaml

from llm_relay.api.app import create_app
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.routing.router import (
    DEFAULT_NON_STREAM_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    SAFETY_SENSITIVE_PARAMS,
    _apply_repetition_penalty_default,
    _clamp_max_tokens,
    _effective_max_tokens,
    _sync_max_tokens_fields,
)
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.router import RequestRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    tmp_path: Path,
    *,
    context_window: int = 262144,
    set_params: dict | None = None,
    strip_params: list | None = None,
    policy: dict | None = None,
) -> Path:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {
            "type": "openai", "base_url": "http://127.0.0.1",
            "ownership": "ciq_owned", "enabled": True,
        }}
    }))
    model_entry: dict = {
        "provider": "local-llm", "class": "unknown",
        "privacy": "local_only", "port": 8080,
        "context_window": context_window,
        "use_cases": {"main": 1},
    }
    if set_params:
        model_entry["set_params"] = set_params
    if strip_params:
        model_entry["strip_params"] = strip_params
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(
        {"models": {"test-model": model_entry}}
    ))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": policy or {}}
    ))
    return cfg_dir


def _plant_client(app) -> None:
    disc = app.state.discovery
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=EndpointState(
            provider="local-llm", status=EndpointStatus.healthy,
            models=["test-model"],
        ),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["test-model"] = "local-llm:8080"


async def _run_through(
    router, monkeypatch, request_data: dict, stream: bool
) -> dict:
    """Send through ``route_and_forward`` capturing what the mock upstream sees."""

    captured: dict = {}

    if stream:
        async def _fake_body_iter():
            yield b"data: {}\n\ndata: [DONE]\n\n"

        async def _fake_cleanup():
            return None

        async def _fake_stream(backend_url, model_name, rd, *a, **kw):
            captured.update(rd)
            return httpx.Response(200, content=b""), _fake_body_iter(), _fake_cleanup

        monkeypatch.setattr(router, "stream_request", _fake_stream)
        result = await router.route_and_forward(
            request_data={"model": "test-model", **request_data},
            stream=True,
        )
        assert result[0].status_code == 200
    else:
        async def _fake_forward(backend_url, model_name, rd, *a, **kw):
            captured.update(rd)
            return httpx.Response(200, json={"choices": []})

        monkeypatch.setattr(router, "forward_request", _fake_forward)
        resp, _ = await router.route_and_forward(
            request_data={"model": "test-model", **request_data},
            stream=False,
        )
        assert resp.status_code == 200

    return captured


# ===========================================================================
# Layer 1: unit tests for new helpers
# ===========================================================================

class TestEffectiveMaxTokens:
    """``_effective_max_tokens`` reads the ceiling from whichever field the
    client used (``max_tokens`` or ``max_completion_tokens``)."""

    def test_reads_max_tokens(self):
        assert _effective_max_tokens({"max_tokens": 1024}) == 1024

    def test_reads_max_completion_tokens(self):
        assert _effective_max_tokens({"max_completion_tokens": 2048}) == 2048

    def test_both_present_takes_stricter(self):
        assert _effective_max_tokens(
            {"max_tokens": 100, "max_completion_tokens": 200}
        ) == 100
        assert _effective_max_tokens(
            {"max_tokens": 200, "max_completion_tokens": 100}
        ) == 100

    def test_neither_present(self):
        assert _effective_max_tokens({"messages": []}) is None

    def test_ignores_zero_and_negative(self):
        assert _effective_max_tokens({"max_tokens": 0}) is None
        assert _effective_max_tokens({"max_tokens": -1}) is None

    def test_valid_among_invalid(self):
        assert _effective_max_tokens(
            {"max_completion_tokens": 0, "max_tokens": 512}
        ) == 512


class TestSyncMaxTokensFields:
    """``_sync_max_tokens_fields`` writes the clamped value back to BOTH
    token-ceiling fields so the upstream sees one consistent number."""

    def test_syncs_both(self):
        out = _sync_max_tokens_fields(
            {"max_tokens": 999, "max_completion_tokens": 999}, 100
        )
        assert out["max_tokens"] == 100
        assert out["max_completion_tokens"] == 100

    def test_only_sets_max_tokens_when_mct_absent(self):
        out = _sync_max_tokens_fields({"max_tokens": 999}, 100)
        assert out["max_tokens"] == 100
        assert "max_completion_tokens" not in out


# ===========================================================================
# Layer 2: _clamp_max_tokens handles max_completion_tokens equivalently
# ===========================================================================

class TestClampWithMaxCompletionTokens:
    """Gap #1: ``max_completion_tokens`` (OpenAI Nov 2024 rename) must be
    treated exactly like ``max_tokens`` — it's the same ceiling with a new
    field name. Without this, a client sending only ``max_completion_tokens``
    would bypass the output budget."""

    def test_caps_max_completion_tokens_alone(self):
        out = _clamp_max_tokens(
            {"max_completion_tokens": 32768, "messages": []}, 10000, 16000
        )
        assert out["max_tokens"] == 6000   # headroom
        assert out["max_completion_tokens"] == 6000  # synced

    def test_stricter_of_both_governs(self):
        out = _clamp_max_tokens(
            {"max_tokens": 100, "max_completion_tokens": 32768, "messages": []},
            10000, 16000,
        )
        # min(100, 32768)=100 fits in 6000 headroom -> no clamp
        assert out["max_tokens"] == 100

    def test_default_applies_via_mct(self):
        out = _clamp_max_tokens(
            {"max_completion_tokens": None, "messages": []}, 10000, 262144,
            default=DEFAULT_NON_STREAM_MAX_TOKENS,
        )
        assert out["max_tokens"] == DEFAULT_NON_STREAM_MAX_TOKENS

    def test_default_caps_max_completion_tokens_only(self):
        """Only max_completion_tokens present + no max_tokens → default still
        applies via the effective ceiling."""
        out = _clamp_max_tokens(
            {"max_completion_tokens": 32768, "messages": []}, 15500, 16000,
            default=DEFAULT_NON_STREAM_MAX_TOKENS,
        )
        assert out["max_completion_tokens"] == 500  # headroom
        assert out["max_tokens"] == 500


# ===========================================================================
# Layer 3: repetition_penalty is a SAFETY FLOOR (not just a default)
# ===========================================================================

class TestRepetitionPenaltyFloor:
    """Gap audit finding: clients that serialize 1.0 as their default bypass
    a presence-only default. The floor clamps ANYTHING below 1.1 up to 1.1."""

    def test_absent_gets_default(self):
        out = _apply_repetition_penalty_default({}, default=1.1)
        assert out["repetition_penalty"] == 1.1

    def test_explicit_1_0_clamped_up(self):
        out = _apply_repetition_penalty_default(
            {"repetition_penalty": 1.0}, default=1.1
        )
        assert out["repetition_penalty"] == 1.1

    def test_explicit_high_left_alone(self):
        src = {"repetition_penalty": 1.2}
        out = _apply_repetition_penalty_default(src, default=1.1)
        assert out is src  # same object, untouched

    def test_null_clamped_up(self):
        out = _apply_repetition_penalty_default(
            {"repetition_penalty": None}, default=1.1
        )
        assert out["repetition_penalty"] == 1.1

    def test_no_default_no_change(self):
        src = {"repetition_penalty": 1.0}
        assert _apply_repetition_penalty_default(src, default=None) is src
        assert _apply_repetition_penalty_default(src, default=0) is src


# ===========================================================================
# Layer 4: set_params cannot override safety defaults
# ===========================================================================

class TestSetParamsSafetyEnforcement:
    """Gap #2: a model config with ``set_params: {repetition_penalty: 1.0}``
    must NOT re-enable the repetition-loop hang. ``_apply_filters``
    re-enforces the floor AFTER applying ``set_params``."""

    def _router(self, tmp_path, set_params):
        cfg_dir = _make_cfg(tmp_path, set_params=set_params, context_window=16000)
        app = create_app(config_dir=cfg_dir)
        return app.state.router

    def test_set_params_rp_belowFloor_is_clamped_back(self, tmp_path):
        router = self._router(tmp_path, {"repetition_penalty": 1.0})
        out = router._apply_filters(
            {"messages": [], "repetition_penalty": 1.2}, "test-model"
        )
        # set_params wrote 1.0, but the floor re-enforced it to 1.1.
        assert out["repetition_penalty"] >= 1.1

    def test_set_params_rp_missing_still_gets_floor(self, tmp_path):
        router = self._router(tmp_path, {"temperature": 0.0})
        out = router._apply_filters({"messages": []}, "test-model")
        assert out["repetition_penalty"] == 1.1
        assert out["temperature"] == 0.0

    def test_set_params_max_tokens_above_window_capped(self, tmp_path):
        router = self._router(tmp_path, {"max_tokens": 999999})
        out = router._apply_filters({"messages": []}, "test-model")
        assert out["max_tokens"] <= 16000  # capped to context_window

    def test_set_params_low_max_tokens_preserved(self, tmp_path):
        """A legitimate low ceiling in set_params IS preserved — we only cap
        ABOVE the context window, never strip all max_tokens from set_params."""
        router = self._router(tmp_path, {"max_tokens": 256})
        out = router._apply_filters({"messages": []}, "test-model")
        assert out["max_tokens"] == 256

    def test_set_params_max_completion_tokens_capped(self, tmp_path):
        router = self._router(tmp_path, {"max_completion_tokens": 999999})
        out = router._apply_filters({"messages": []}, "test-model")
        assert out.get("max_completion_tokens", 999999) <= 16000

    def test_normal_set_params_still_works(self, tmp_path):
        """Non-safety set_params (temperature, top_p) pass through unmolested."""
        router = self._router(tmp_path, {"temperature": 0.0, "top_p": 0.9})
        out = router._apply_filters(
            {"messages": [], "temperature": 0.7}, "test-model"
        )
        assert out["temperature"] == 0.0
        assert out["top_p"] == 0.9
        assert out["repetition_penalty"] == 1.1  # floor still added


# ===========================================================================
# Layer 5: config-loader warns about safety-sensitive set_params
# ===========================================================================

class TestConfigLoaderWarning:
    """Gap #2 (config side): loading a model with safety-sensitive fields in
    ``set_params`` emits a visible WARNING so the operator knows the runtime
    will override them."""

    def test_warning_emitted_for_rp_in_set_params(self, tmp_path, caplog):
        cfg_dir = _make_cfg(tmp_path, set_params={"repetition_penalty": 1.0})
        with caplog.at_level(logging.WARNING, logger="llm_relay.config"):
            create_app(config_dir=cfg_dir)
        joined = " ".join(r.message for r in caplog.records)
        assert "test-model" in joined
        assert "safety-sensitive" in joined
        assert "repetition_penalty" in joined

    def test_warning_emitted_for_max_tokens_in_set_params(self, tmp_path, caplog):
        cfg_dir = _make_cfg(tmp_path, set_params={"max_tokens": 999999})
        with caplog.at_level(logging.WARNING, logger="llm_relay.config"):
            create_app(config_dir=cfg_dir)
        joined = " ".join(r.message for r in caplog.records)
        assert "safety-sensitive" in joined
        assert "max_tokens" in joined

    def test_warning_emitted_for_max_completion_tokens(self, tmp_path, caplog):
        cfg_dir = _make_cfg(tmp_path, set_params={"max_completion_tokens": 999999})
        with caplog.at_level(logging.WARNING, logger="llm_relay.config"):
            create_app(config_dir=cfg_dir)
        joined = " ".join(r.message for r in caplog.records)
        assert "safety-sensitive" in joined
        assert "max_completion_tokens" in joined

    def test_no_warning_for_safe_set_params(self, tmp_path, caplog):
        cfg_dir = _make_cfg(tmp_path, set_params={"temperature": 0.0})
        with caplog.at_level(logging.WARNING, logger="llm_relay.config"):
            create_app(config_dir=cfg_dir)
        assert not any("safety-sensitive" in r.message for r in caplog.records)


# ===========================================================================
# Layer 6: the full conformance matrix
# ===========================================================================
#
# {streaming, non-streaming} x {max_tokens variants} x {rep_penalty variants}
#
# Each cell runs the SAME payload through BOTH paths and asserts the forwarded
# safety fields are IDENTICAL. This is the test that would have caught the
# streaming-path omission before it shipped.

_MT_VARIANTS = [
    ("absent", {}, DEFAULT_NON_STREAM_MAX_TOKENS),
    ("huge", {"max_tokens": 32768}, 32768),
    ("safe", {"max_tokens": 256}, 256),
    ("mct_only", {"max_completion_tokens": 32768}, 32768),
    ("both_mixed", {"max_tokens": 100, "max_completion_tokens": 32768}, 100),
]

_RP_VARIANTS = [
    ("absent", {}),
    ("explicit_1_0", {"repetition_penalty": 1.0}),
    ("explicit_1_2", {"repetition_penalty": 1.2}),
]


def _matrix_ids():
    ids = []
    for mt_label, _, _ in _MT_VARIANTS:
        for rp_label, _ in _RP_VARIANTS:
            ids.append(f"mt_{mt_label}|rp_{rp_label}")
    return ids


def _matrix_payloads():
    payloads = []
    for _, mt_payload, _ in _MT_VARIANTS:
        for _, rp_payload in _RP_VARIANTS:
            payloads.append({**mt_payload, **rp_payload})
    return payloads


@pytest.mark.parametrize("payload", _matrix_payloads(), ids=_matrix_ids())
async def test_streaming_and_non_streaming_produce_identical_defaults(
    tmp_path, monkeypatch, payload
):
    """Every parameter combination MUST produce the same forwarded max_tokens
    and repetition_penalty on BOTH paths. This is the core conformance
    invariant: the ``stream`` flag is a response-framing choice, not a
    safety-bypass switch."""
    # Fresh app per cell so monkeypatch state can't bleed.
    cfg_dir = _make_cfg(tmp_path)
    app = create_app(config_dir=cfg_dir)
    _plant_client(app)
    router = app.state.router

    cap_ns = await _run_through(router, monkeypatch, dict(payload), stream=False)
    cap_stream = await _run_through(router, monkeypatch, dict(payload), stream=True)

    # The two paths MUST agree on what reaches the upstream.
    assert cap_ns.get("max_tokens") == cap_stream.get("max_tokens"), (
        f"path divergence: non-stream={cap_ns.get('max_tokens')} "
        f"stream={cap_stream.get('max_tokens')} for payload={payload}"
    )
    assert cap_ns.get("repetition_penalty") == cap_stream.get("repetition_penalty"), (
        f"path divergence: non-stream={cap_ns.get('repetition_penalty')} "
        f"stream={cap_stream.get('repetition_penalty')} for payload={payload}"
    )

    # Assert specific safety properties that hold on BOTH paths:
    rp = cap_ns.get("repetition_penalty")
    assert rp is not None, "repetition_penalty must always be set (floor)"
    assert rp >= DEFAULT_REPETITION_PENALTY, (
        f"repetition_penalty {rp} must be >= floor {DEFAULT_REPETITION_PENALTY}"
    )

    # max_tokens must be present on BOTH paths, sourced from the client's
    # ceiling (whichever field name), the default (if absent), or the
    # headroom clamp (if the ceiling exceeded it). The effective ceiling is
    # the MIN of any token-limit field the client set.
    expected_ceiling = _effective_max_tokens(payload)
    if expected_ceiling is None:
        expected_ceiling = DEFAULT_NON_STREAM_MAX_TOKENS
    mt = cap_ns.get("max_tokens")
    assert mt is not None, "max_tokens must always be set (default or client)"
    assert mt <= expected_ceiling, (
        f"max_tokens {mt} must be <= the effective ceiling {expected_ceiling}; "
        f"a headroom clamp can only LOWER it"
    )


@pytest.mark.parametrize("payload", _matrix_payloads(), ids=_matrix_ids())
async def test_max_completion_tokens_is_canonicalized_on_both_paths(
    tmp_path, monkeypatch, payload
):
    """When the client sends ``max_completion_tokens`` (with or without
    ``max_tokens``), the forwarded request must have ``max_tokens`` set to
    the effective ceiling on BOTH paths."""
    if "max_completion_tokens" not in payload:
        pytest.skip("payload doesn't use max_completion_tokens")

    cfg_dir = _make_cfg(tmp_path)
    app = create_app(config_dir=cfg_dir)
    _plant_client(app)
    router = app.state.router

    cap_ns = await _run_through(router, monkeypatch, dict(payload), stream=False)
    cap_stream = await _run_through(router, monkeypatch, dict(payload), stream=True)

    for label, cap in [("non-stream", cap_ns), ("stream", cap_stream)]:
        assert "max_tokens" in cap, (
            f"{label}: max_tokens must be set even when client used "
            f"max_completion_tokens; got keys={list(cap.keys())}"
        )
