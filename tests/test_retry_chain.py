"""Verify route_and_forward walks the fallback chain on retry_on errors."""
from __future__ import annotations

from pathlib import Path
import httpx
import pytest
import yaml

from llm_relay.api.app import create_app
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus, SaturationError
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.router import RequestRouter


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> Path:
    """Write a two-model config where alias 'main' resolves to [model-a, model-b]."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {
                "type": "openai",
                "base_url": "http://127.0.0.1",
                "enabled": True,
            }
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "model-a": {
                "provider": "local-llm",
                "class": "unknown",
                "privacy": "local_only",
                "port": 8080,
            },
            "model-b": {
                "provider": "local-llm",
                "class": "unknown",
                "privacy": "local_only",
                "port": 8081,
            },
        },
        "aliases": {
            "main": ["model-a", "model-b"],
        },
    }))
    # Minimal policy — no fallback graph, default retry_on
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump({
        "policy": {
            "fallback": {
                "retry_on": ["502", "503", "504", "connection_error"],
            }
        }
    }))
    return cfg_dir


def _make_app_with_both_healthy(tmp_path: Path):
    """Return a (app, router) pair with model-a and model-b both available."""
    cfg_dir = _make_config(tmp_path)
    app = create_app(config_dir=cfg_dir)

    disc = app.state.discovery
    # Manually plant both clients so the selector sees them as available.
    state_a = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["model-a"])
    state_b = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["model-b"])
    disc.clients["local-llm:8080"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8080",
        state=state_a, circuit_breaker=CircuitBreaker(),
    )
    disc.clients["local-llm:8081"] = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:8081",
        state=state_b, circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client["model-a"] = "local-llm:8080"
    disc.model_to_client["model-b"] = "local-llm:8081"

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_route_and_forward_retries_next_candidate_on_502(tmp_path, monkeypatch):
    """Backend A returns 502 → router tries backend B → 200 succeeds."""
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    resp_502 = httpx.Response(502, content=b"bad gateway")
    resp_200 = httpx.Response(200, json={"choices": []})

    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return resp_502
        return resp_200

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "main", "messages": []},
        stream=False,
    )

    assert resp.status_code == 200
    assert call_count == 2, "Should have tried model-a (502) then model-b (200)"
    # Telemetry should reflect the winning candidate (model-b)
    assert result.selected_model == "model-b"
    assert result.success is True


async def test_route_and_forward_retries_next_candidate_on_connection_error(tmp_path, monkeypatch):
    """Backend A raises ConnectError → router tries backend B → 200."""
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    resp_200 = httpx.Response(200, json={"choices": []})
    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("connection refused")
        return resp_200

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "main", "messages": []},
        stream=False,
    )

    assert resp.status_code == 200
    assert call_count == 2
    assert result.selected_model == "model-b"


async def test_route_and_forward_propagates_non_retryable_400(tmp_path, monkeypatch):
    """Backend A returns 400 → router does NOT retry; 400 surfaces."""
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"error": "bad request"})

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "main", "messages": []},
        stream=False,
    )

    assert resp.status_code == 400
    assert call_count == 1, "Should NOT retry on 400"


async def test_route_and_forward_exhausts_chain_returns_last_502(tmp_path, monkeypatch):
    """Every candidate returns 502 → router returns the last 502."""
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return httpx.Response(502, content=b"bad gateway")

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "main", "messages": []},
        stream=False,
    )

    assert resp.status_code == 502
    assert call_count == 2, "Should have tried both candidates before giving up"
    # RouteResult should reflect the last candidate attempted (model-b)
    assert result.selected_model == "model-b"


async def test_route_and_forward_propagates_saturation_error_immediately(tmp_path, monkeypatch):
    """SaturationError from backend A does NOT trigger retry to backend B.

    Slot saturation is backpressure, not backend failure — surface immediately
    so the caller can back off (Retry-After) instead of cascading load.

    Critically: forward_request is called EXACTLY ONCE when SaturationError fires.
    This asserts behavioral correctness (not just status code), ensuring the
    retry loop does not silently swallow and re-raise the exception.
    """
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise SaturationError(backend_key="local-llm:8080", retry_after_seconds=3.0)

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    with pytest.raises(SaturationError) as excinfo:
        await router.route_and_forward(
            request_data={"model": "main", "messages": []},
            stream=False,
        )

    # Must have stopped after the first backend — NOT tried model-b.
    assert call_count == 1, (
        "SaturationError must NOT trigger cross-backend retry; "
        f"forward_request was called {call_count} time(s)"
    )
    assert excinfo.value.backend_key == "local-llm:8080"
    assert excinfo.value.retry_after_seconds == 3.0


async def test_route_and_forward_503_via_api_on_saturation(tmp_path, monkeypatch):
    """End-to-end: SaturationError still produces 503 + Retry-After at the HTTP layer."""
    app = _make_app_with_both_healthy(tmp_path)

    async def _fake_forward(*args, **kwargs):
        raise SaturationError(backend_key="local-llm:8080", retry_after_seconds=4.0)

    monkeypatch.setattr(app.state.router, "forward_request", _fake_forward)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "main", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 503
    assert "Retry-After" in resp.headers
    assert int(resp.headers["Retry-After"]) >= 1
    body = resp.json()
    assert body["detail"]["error"] == "backend saturated"


async def test_route_and_forward_streaming_passes_through_response(tmp_path, monkeypatch):
    """Streaming path returns a 3-tuple (resp, iterator, route_result) from the first
    available candidate without retrying across backends.

    Verifies the app.py 3-tuple unpack and that result.selected_model reflects
    the actual candidate that was used.
    """
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    sse_bytes = b"data: {}\n\ndata: [DONE]\n\n"

    async def _fake_body_iter():
        yield sse_bytes

    async def _fake_stream(backend_url, model_name, *args, **kwargs):
        return httpx.Response(200, content=b""), _fake_body_iter()

    monkeypatch.setattr(router, "stream_request", _fake_stream)

    upstream, body_iter, result = await router.route_and_forward(
        request_data={"model": "main", "messages": [], "stream": True},
        stream=True,
    )

    assert upstream.status_code == 200
    # Collect all bytes from the iterator
    chunks = []
    async for chunk in body_iter:
        chunks.append(chunk)
    assert b"DONE" in b"".join(chunks)
    # Should have picked model-a (first in alias order) since both healthy
    assert result.selected_model == "model-a"
    assert result.success is True


async def test_route_and_forward_exhausted_chain_reports_correct_candidate(tmp_path, monkeypatch):
    """When chain exhausts via 502-then-network-error, the returned RouteResult
    must describe the candidate that produced the response (not the last attempted).

    Regression for the telemetry mismatch where candidates[-1] was used instead
    of the candidate that actually returned the held-onto response.
    """
    app = _make_app_with_both_healthy(tmp_path)
    router = app.state.router

    call_count = 0

    async def _fake_forward(backend_url, model_name, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Candidate 0 (model-a) returns 502 — retryable, becomes last_response.
            return httpx.Response(502, content=b"bad gateway")
        # Candidate 1 (model-b) raises a network exception — sets last_error,
        # but leaves last_response pointing at the model-a 502 response.
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(router, "forward_request", _fake_forward)

    resp, result = await router.route_and_forward(
        request_data={"model": "main", "messages": []},
        stream=False,
    )

    assert call_count == 2
    assert resp.status_code == 502
    # The reported candidate must be model-a (which served the 502 response),
    # NOT model-b (the last attempted, which network-errored).
    assert result.selected_model == "model-a", (
        f"expected model-a (which served the 502 response), got {result.selected_model}"
    )
