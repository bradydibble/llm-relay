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
    MIN_OUTPUT_HEADROOM,
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
    assert _clamp_max_tokens(no_mt, 10000, 16000) is no_mt   # no ceiling to clamp
    leave = {"max_tokens": 32768}
    assert _clamp_max_tokens(leave, 0, 16000) is leave       # trivially-small prompt -> leave
    assert _clamp_max_tokens(leave, 10000, 0) is leave       # unknown window -> leave


# --- reasoning floor: raise a too-small ceiling for reasoning models ----------

def test_clamp_floors_small_max_tokens_for_reasoning_model():
    # Tiny client ceiling on a reasoning model: reasoning would consume it all, so
    # floor up to the reasoning floor (well under the 262k window headroom).
    out = _clamp_max_tokens({"max_tokens": 400, "messages": []}, 3750, 262144, reasoning_floor=2048)
    assert out["max_tokens"] == 2048


def test_clamp_floor_never_exceeds_headroom():
    # Floor is capped by the window headroom — fit always wins over floor.
    out = _clamp_max_tokens({"max_tokens": 100, "messages": []}, 15500, 16000, reasoning_floor=2048)
    assert out["max_tokens"] == 500  # headroom 16000 - 15500, not 2048


def test_clamp_floor_noop_when_already_above():
    src = {"max_tokens": 4000, "messages": []}
    # already above the floor and fits the window -> unchanged, same object.
    assert _clamp_max_tokens(src, 3750, 262144, reasoning_floor=2048) is src


def test_clamp_floor_zero_is_legacy_downclamp_only():
    # reasoning_floor=0 (every non-reasoning model) -> pure down-clamp, no floor.
    src = {"max_tokens": 400, "messages": []}
    assert _clamp_max_tokens(src, 3750, 262144, reasoning_floor=0) is src


# --- integration: a big max_tokens no longer pins to the largest backend ------

def _make_cfg(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True}}
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


async def test_reasoning_model_floors_small_max_tokens(tmp_path, monkeypatch):
    """A model advertising the `reasoning` capability floors a too-small client
    max_tokens up to REASONING_OUTPUT_FLOOR, so the chain-of-thought cannot
    consume the whole ceiling and leave the answer empty (the ornith-397b failure
    mode). Proves the capability gating + call-site wiring, not just the helper."""
    from llm_relay.routing.router import REASONING_OUTPUT_FLOOR

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True}}
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
    assert captured["max_tokens"] == REASONING_OUTPUT_FLOOR, "small ceiling floored for reasoning model"
