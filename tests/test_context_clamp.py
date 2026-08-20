"""max_tokens is an output ceiling, not a routing reservation.

Eligibility is sized on the PROMPT (plus a small output floor); the forwarded
max_tokens is CLAMPED to the chosen model's headroom. So a generous max_tokens
no longer pins a request to the single largest-context backend (open-by-default
degradation is restored), and a vLLM backend never receives
prompt + max_tokens > max_model_len (which it hard-rejects with a 400).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import yaml

from llm_relay.api.app import create_app
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.routing.router import (
    DEFAULT_NON_STREAM_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    MIN_OUTPUT_HEADROOM,
    _apply_repetition_penalty_default,
    _clamp_max_tokens,
    _estimate_prompt_tokens,
)


# --- _estimate_prompt_tokens: prompt only, never the output ceiling ----------

def test_estimate_prompt_tokens_counts_prompt_not_max_tokens():
    # tiktoken (cl100k_base) exact-counts the prompt: 3000 repeated "x" BPE-pack
    # to 375 tokens; max_tokens (an output ceiling) is IGNORED either way.
    assert _estimate_prompt_tokens(
        {"messages": [{"role": "user", "content": "x" * 3000}], "max_tokens": 50000}
    ) == 375


# --- _clamp_max_tokens: cap the output to the chosen model's headroom ----------

def test_clamp_caps_max_tokens_to_headroom():
    # window 16000, prompt 10000 -> headroom 6000; a 32768 ceiling is clamped down.
    out = _clamp_max_tokens({"max_tokens": 32768, "messages": []}, 10000, 16000)
    assert out["max_tokens"] == 6000


def test_clamp_leaves_request_that_already_fits():
    src = {"max_tokens": 4000, "messages": []}
    # 4000 fits in 65536 - 10000 -> returned unchanged, no copy.
    assert _clamp_max_tokens(src, 10000, 65536) is src


def test_clamp_does_not_mutate_caller_dict():
    src = {"max_tokens": 32768}
    _clamp_max_tokens(src, 60000, 65536)
    assert src["max_tokens"] == 32768  # caller's dict preserved; clamp works on a copy


def test_clamp_noops_without_max_tokens_or_window():
    no_mt = {"messages": []}
    # no max_tokens, no default -> unchanged (streaming path: client sees progress)
    assert _clamp_max_tokens(no_mt, 10000, 16000) is no_mt
    leave = {"max_tokens": 32768}
    assert _clamp_max_tokens(leave, 0, 16000) is leave       # trivially-small prompt -> leave
    assert _clamp_max_tokens(leave, 10000, 0) is leave       # unknown window -> leave


# --- default max_tokens for non-streaming (unbounded-generation protection) ---

def test_clamp_applies_default_when_no_max_tokens():
    """A non-streaming request with no max_tokens gets a default cap so vLLM
    doesn't generate for hours (262K context default at 10 tok/s = ~7h).
    The default is applied, then capped to headroom if needed."""
    out = _clamp_max_tokens({"messages": []}, 10000, 262144,
                            default=DEFAULT_NON_STREAM_MAX_TOKENS)
    assert out["max_tokens"] == DEFAULT_NON_STREAM_MAX_TOKENS


def test_clamp_default_capped_to_headroom():
    """The default itself is capped to the model's headroom — a small-context
    model gets a smaller effective cap."""
    out = _clamp_max_tokens({"messages": []}, 15500, 16000,
                            default=DEFAULT_NON_STREAM_MAX_TOKENS)
    assert out["max_tokens"] == 500  # 16000 - 15500 < default


def test_clamp_default_does_not_override_client_ceiling():
    """A client-set max_tokens is NEVER raised toward the default. The default
    only applies when the client set NOTHING. This is NOT the removed
    REASONING_OUTPUT_FLOOR (which inflated 16 -> 2048)."""
    src = {"max_tokens": 16, "messages": []}
    # client asked for 16; default is 1024; forwarded value is 16, NOT 1024
    out = _clamp_max_tokens(src, 3750, 262144, default=DEFAULT_NON_STREAM_MAX_TOKENS)
    assert out is src  # same object, unchanged — client ceiling untouched


def test_clamp_default_not_applied_for_streaming():
    """Streaming path passes no default — the client sees tokens flowing and
    can disconnect, so unbounded generation is the client's choice."""
    src = {"messages": []}
    out = _clamp_max_tokens(src, 10000, 16000)  # no default= kwarg
    assert out is src  # unchanged


# --- the ceiling is the client's: cap down, NEVER inflate ---------------------
# REASONING_OUTPUT_FLOOR was removed 2026-08-11 (see the tombstone in router.py):
# it silently rewrote a client's cost ceiling (asked 16, forwarded 2048 -
# measured). Reasoning is separated at the source now, so a small ceiling gives
# the honest OpenAI-style outcome instead: finish_reason=length, reasoning in
# reasoning_content, content possibly empty. These tests pin the ceiling as
# untouchable so the floor cannot quietly return.

def test_clamp_never_raises_small_max_tokens():
    src = {"max_tokens": 16, "messages": []}
    # Plenty of headroom: the tiny ceiling is the client's choice; forwarded as-is.
    assert _clamp_max_tokens(src, 3750, 262144) is src


def test_clamp_caps_down_to_headroom():
    out = _clamp_max_tokens({"max_tokens": 4000, "messages": []}, 15500, 16000)
    assert out["max_tokens"] == 500  # 16000 - 15500; fit still wins


def test_clamp_noop_returns_same_object():
    src = {"max_tokens": 400, "messages": []}
    assert _clamp_max_tokens(src, 3750, 262144) is src


# --- integration: a big max_tokens no longer pins to the largest backend ------

def _make_cfg(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            # `main` is derived from use_cases tags: big-model preferred, small-model fallback.
            "big-model": {"provider": "local-llm", "class": "unknown",
                          "privacy": "local_only", "port": 8080, "context_window": 100000,
                          "use_cases": {"main": 2}},
            "small-model": {"provider": "local-llm", "class": "unknown",
                            "privacy": "local_only", "port": 8081, "context_window": 16000,
                            "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump({
        "policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]}}
    }))
    return cfg_dir


async def test_big_max_tokens_degrades_to_smaller_model_with_clamp(tmp_path, monkeypatch):
    """big-model down; a 10k-token prompt + 32768 max_tokens must still be served
    by small-model (16k window) — eligibility is sized on the prompt — with the
    forwarded max_tokens clamped to small-model's headroom (16000 - 10000 = 6000).

    Pre-fix this 503'd: prompt + max_tokens (42768) excluded the 16k model, and the
    only model big enough (big-model) was down -> the open-fallthrough dead-ended.
    """
    cfg_dir = _make_cfg(tmp_path)
    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    # Only small-model is live; big-model is unplanted -> unavailable (down).
    disc.clients["local-llm:8081"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8081",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["small-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["small-model"] = "local-llm:8081"
    router = app.state.router

    captured: dict = {}

    async def _fake_forward(backend_url, model_name, request_data, *args, **kwargs):
        captured["model"] = model_name
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={
            "model": "main",
            "messages": [{"role": "user", "content": "x" * 30000}],  # 3750 tiktoken tokens
            "max_tokens": 32768,
        },
        stream=False,
    )

    assert resp.status_code == 200
    assert captured["model"] == "small-model", "request must degrade to the live small model"
    assert captured["max_tokens"] == 12250, "output clamped to small-model headroom (16000 - 3750)"
    assert MIN_OUTPUT_HEADROOM > 0


async def test_big_max_tokens_degrades_with_clamp_streaming(tmp_path, monkeypatch):
    """Same degrade-and-clamp on the SSE path: the streaming loop clamps the
    forwarded max_tokens to the chosen model's headroom too (a primary workload,
    and a separate code path from the non-streaming loop above)."""
    cfg_dir = _make_cfg(tmp_path)
    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8081"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8081",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["small-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["small-model"] = "local-llm:8081"
    router = app.state.router

    captured: dict = {}

    async def _fake_body_iter():
        yield b"data: {}\n\ndata: [DONE]\n\n"

    async def _fake_cleanup():
        return None

    async def _fake_stream(backend_url, model_name, request_data, *args, **kwargs):
        captured["model"] = model_name
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, content=b""), _fake_body_iter(), _fake_cleanup

    monkeypatch.setattr(router, "stream_request", _fake_stream)

    upstream, body_iter, result, cleanup = await router.route_and_forward(
        request_data={
            "model": "main",
            "messages": [{"role": "user", "content": "x" * 30000}],  # 3750 tiktoken tokens
            "max_tokens": 32768,
            "stream": True,
        },
        stream=True,
    )

    assert upstream.status_code == 200
    assert captured["model"] == "small-model"
    assert captured["max_tokens"] == 12250, "output clamped on the streaming path too (16000 - 3750)"


async def test_reasoning_model_gets_client_ceiling_unchanged(tmp_path, monkeypatch):
    """A reasoning model receives the client's max_tokens UNCHANGED. The old
    floor inflated it to REASONING_OUTPUT_FLOOR; that rewrite of the client's
    cost ceiling was removed 2026-08-11 (reasoning is separated at the serve,
    so a starved answer is now an honest finish_reason=length, not a mystery).
    Proves the call-site wiring end to end, not just the helper."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "reasoner": {"provider": "local-llm", "class": "unknown", "privacy": "local_only",
                         "port": 8082, "context_window": 262144,
                         "capabilities": ["reasoning"], "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]}}}
    ))

    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8082"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8082",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["reasoner"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["reasoner"] = "local-llm:8082"
    router = app.state.router

    captured: dict = {}

    async def _fake_forward(backend_url, model_name, request_data, *args, **kwargs):
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "reasoner",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 400},
        stream=False,
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] == 400, "the client's ceiling reaches the backend untouched"


async def test_non_stream_no_max_tokens_gets_default(tmp_path, monkeypatch):
    """A non-streaming request with NO max_tokens must get a default cap
    forwarded to the backend. Without this, vLLM defaults to max_model_len -
    prompt (~247K tokens = ~7 hours at 10 tok/s on qwen3.6-35b), the relay
    buffers the whole response, and the client gets nothing but keepalive
    whitespace — the 2026-08-16 narf-agent indefinite-hang incident."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1",
                         "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "big-model": {"provider": "local-llm", "class": "unknown",
                          "privacy": "local_only", "port": 8080, "context_window": 262144,
                          "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]}}}
    ))

    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["big-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["big-model"] = "local-llm:8080"
    router = app.state.router

    captured: dict = {}

    async def _fake_forward(backend_url, model_name, request_data, *args, **kwargs):
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "big-model",
                      "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] == DEFAULT_NON_STREAM_MAX_TOKENS, \
        "non-streaming without max_tokens must get the default cap, not vLLM's unbounded default"


async def test_stream_no_max_tokens_gets_no_default(tmp_path, monkeypatch):
    """Streaming without max_tokens must NOT get a default — the client sees
    tokens flowing and can disconnect; the default is a non-streaming protection
    only (prevents the buffered-response hang)."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1",
                         "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "big-model": {"provider": "local-llm", "class": "unknown",
                          "privacy": "local_only", "port": 8080, "context_window": 262144,
                          "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]}}}
    ))

    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["big-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["big-model"] = "local-llm:8080"
    router = app.state.router

    captured: dict = {}

    async def _fake_body_iter():
        yield b"data: {}\n\ndata: [DONE]\n\n"

    async def _fake_cleanup():
        return None

    async def _fake_stream(backend_url, model_name, request_data, *args, **kwargs):
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, content=b""), _fake_body_iter(), _fake_cleanup

    monkeypatch.setattr(router, "stream_request", _fake_stream)

    upstream, body_iter, result, cleanup = await router.route_and_forward(
        request_data={"model": "big-model",
                      "messages": [{"role": "user", "content": "hi"}],
                      "stream": True},
        stream=True,
    )
    assert upstream.status_code == 200
    assert captured["max_tokens"] is None, \
        "streaming without max_tokens must NOT get a default — client controls duration"


async def test_policy_default_max_tokens_override(tmp_path, monkeypatch):
    """policy.yaml can set default_max_tokens to tune the non-streaming cap
    without a code change. Set to 0 to disable (restore unbounded generation)."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1",
                         "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "big-model": {"provider": "local-llm", "class": "unknown",
                          "privacy": "local_only", "port": 8080, "context_window": 262144,
                          "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]},
                     "default_max_tokens": 4096}}
    ))

    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["big-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["big-model"] = "local-llm:8080"
    router = app.state.router

    captured: dict = {}

    async def _fake_forward(backend_url, model_name, request_data, *args, **kwargs):
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "big-model",
                      "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] == 4096, \
        "policy.yaml default_max_tokens override must reach the backend"


async def test_policy_default_max_tokens_disabled(tmp_path, monkeypatch):
    """Setting default_max_tokens: 0 in policy.yaml disables the cap entirely
    (restores unbounded generation for clients that set no max_tokens)."""

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1",
                         "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "big-model": {"provider": "local-llm", "class": "unknown",
                          "privacy": "local_only", "port": 8080, "context_window": 262144,
                          "use_cases": {"main": 1}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]},
                     "default_max_tokens": 0}}
    ))

    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["big-model"]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["big-model"] = "local-llm:8080"
    router = app.state.router

    captured: dict = {}

    async def _fake_forward(backend_url, model_name, request_data, *args, **kwargs):
        captured["max_tokens"] = request_data.get("max_tokens")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "big-model",
                      "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] is None, \
        "default_max_tokens: 0 disables the cap — client's none stays none"


# --- default repetition_penalty for non-streaming (repetition-loop protection) ---

def test_repetition_penalty_applied_when_unset():
    """A non-streaming request with no repetition_penalty gets the default
    (1.1, matching Ollama) so the model doesn't loop forever on certain
    prompts (the 2026-08-16 narf-agent hang: diff-marker C declaration
    triggered an infinite 'int64 vs int64' repetition loop)."""
    out = _apply_repetition_penalty_default({"messages": []}, default=1.1)
    assert out["repetition_penalty"] == 1.1


def test_repetition_penalty_not_overriding_client_value():
    """A client-set repetition_penalty is NEVER overridden — the default only
    applies when the client set nothing."""
    src = {"repetition_penalty": 1.3, "messages": []}
    out = _apply_repetition_penalty_default(src, default=1.1)
    assert out["repetition_penalty"] == 1.3


def test_repetition_penalty_disabled_when_zero():
    """default=0 disables the feature (no penalty applied)."""
    src = {"messages": []}
    out = _apply_repetition_penalty_default(src, default=0)
    assert out is src  # unchanged


def test_repetition_penalty_disabled_when_none():
    """default=None disables the feature."""
    src = {"messages": []}
    out = _apply_repetition_penalty_default(src, default=None)
    assert out is src  # unchanged
