"""Integration tests: the /metrics endpoint and request-path recording.

These exercise the wiring (app mount + emit_chat_completion → record_request)
through the real FastAPI app with a mocked upstream, against the dedicated
RELAY_REGISTRY. Counters accumulate process-wide, so assertions use deltas.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import yaml

from llm_relay.api.app import create_app
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.metrics import RELAY_REGISTRY


def _make_app(tmp_path: Path):
    """App where alias 'main' -> [model-a, model-b], both healthy (model-a preferred)."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "model-a": {"provider": "local-llm", "privacy": "local_only", "port": 8080},
            "model-b": {"provider": "local-llm", "privacy": "local_only", "port": 8081},
            # ConfigLoader reads aliases from a nested `aliases:` key under `models:`.
            "aliases": {"main": ["model-a", "model-b"]},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump({
        "policy": {"fallback": {"retry_on": ["502", "503", "504", "connection_error"]}}
    }))
    app = create_app(config_dir=cfg_dir)
    disc = app.state.discovery
    for model, port in (("model-a", 8080), ("model-b", 8081)):
        key = f"local-llm:{port}"
        disc.clients[key] = EndpointClient(
            provider_name="local-llm", base_url=f"http://127.0.0.1:{port}",
            state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=[model]),
            circuit_breaker=CircuitBreaker(),
        )
        disc.model_to_client[model] = key
    return app


def _val(name, labels):
    return RELAY_REGISTRY.get_sample_value(name, labels) or 0.0


async def test_metrics_endpoint_serves_exposition_with_backend_gauges(tmp_path):
    app = _make_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "llm_relay_backend_up" in resp.text


async def test_successful_request_records_request_and_completion_tokens(tmp_path, monkeypatch):
    app = _make_app(tmp_path)

    async def _fake_forward(backend_url, model_name, *a, **k):
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        })

    monkeypatch.setattr(app.state.router, "forward_request", _fake_forward)

    req_labels = {"provider": "local-llm", "model": "model-a", "alias": "main",
                  "outcome": "success", "client": "claude-code"}
    tok_labels = {"provider": "local-llm", "model": "model-a",
                  "direction": "completion", "client": "claude-code"}
    before_req = _val("llm_relay_requests_total", req_labels)
    before_tok = _val("llm_relay_tokens_total", tok_labels)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "main", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Llm-Relay-Client": "claude-code"},
        )

    assert resp.status_code == 200
    assert _val("llm_relay_requests_total", req_labels) == before_req + 1.0
    assert _val("llm_relay_tokens_total", tok_labels) == before_tok + 9.0


async def test_fallback_request_records_fallbacks_total(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    calls = {"n": 0}

    async def _fake_forward(backend_url, model_name, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, content=b"bad gateway")  # model-a fails
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(app.state.router, "forward_request", _fake_forward)

    fb_labels = {"alias": "main", "model": "model-b", "client": "unknown"}
    before = _val("llm_relay_fallbacks_total", fb_labels)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "main", "messages": []},
        )

    assert resp.status_code == 200
    assert calls["n"] == 2  # fell back model-a -> model-b
    assert _val("llm_relay_fallbacks_total", fb_labels) == before + 1.0
