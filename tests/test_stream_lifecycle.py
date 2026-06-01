"""Streaming slot-lifecycle: the in-flight slot must be released no matter how
the stream ends — normal completion, upstream error, or a client disconnect
that arrives as a CancelledError storm partway through cleanup.

The historic leak: ``stream_request``'s ``_iter()`` finally released the slot
*last*, after two ``await``s (``resp.aclose()`` / ``client.aclose()``). If
cancellation interrupted either await, the slot release was skipped and the
backend's ``inflight_used`` drifted up by one per leaked stream — eventually
phantom-saturating the backend. The fix makes slot release synchronous and
first, so no await can preempt it.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.responses import StreamingResponse as RealStreamingResponse
from httpx import ASGITransport

import llm_relay.api.app as app_mod
from llm_relay.api.app import create_app
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus, SaturationError
from llm_relay.discovery import endpoint as endpoint_mod
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing import router as router_mod
from llm_relay.routing.keys import compose_backend_key
from llm_relay.routing.router import RequestRouter, RouteResult


class _FakeResp:
    """Minimal stand-in for an httpx streaming Response."""

    def __init__(self, *, aclose_raises: BaseException | None = None):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self._aclose_raises = aclose_raises

    async def aiter_raw(self):
        yield b"data: {}\n\n"
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        if self._aclose_raises is not None:
            raise self._aclose_raises


class _FakeClientFactory:
    """Returns a callable usable as a drop-in for ``httpx.AsyncClient(...)``."""

    def __init__(self, resp: _FakeResp):
        self._resp = resp

    def __call__(self, *args, **kwargs):
        resp = self._resp

        class _FakeClient:
            def build_request(self, *a, **k):
                return object()

            async def send(self, req, stream=False):
                return resp

            async def aclose(self):
                pass

        return _FakeClient()


def _bounded_router(monkeypatch, resp: _FakeResp) -> tuple[RequestRouter, EndpointClient]:
    monkeypatch.setattr(router_mod.httpx, "AsyncClient", _FakeClientFactory(resp))
    disc = DiscoveryManager()
    disc.clients["k"] = EndpointClient(
        provider_name="local-llm",
        base_url="http://127.0.0.1:9",
        state=EndpointState(provider="local-llm"),
        circuit_breaker=CircuitBreaker(),
        max_concurrent=1,
    )
    # stream_request never reads self.config; only self.discovery + httpx.
    router = RequestRouter(config=None, discovery=disc)  # type: ignore[arg-type]
    return router, disc.clients["k"]


async def _drain(body_iter) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        async for _ in body_iter:
            pass


async def test_stream_request_releases_slot_when_aclose_raises_cancelled(monkeypatch):
    """If resp.aclose() raises CancelledError mid-cleanup (the disconnect storm),
    the slot must STILL be released. Slot release must not sit behind an await
    that cancellation can preempt."""
    resp = _FakeResp(aclose_raises=asyncio.CancelledError())
    router, client = _bounded_router(monkeypatch, resp)

    ret = await router.stream_request(
        "http://127.0.0.1:9", "m", {"messages": []},
        backend_key="k", slot_wait_timeout=1.0,
    )
    body_iter = ret[1]
    assert client.inflight_used == 1, "slot should be held while the stream is open"

    await _drain(body_iter)

    assert client.inflight_used == 0, (
        "slot leaked: release was skipped when aclose() raised CancelledError"
    )


async def test_stream_request_releases_slot_on_normal_completion(monkeypatch):
    """Happy path regression guard: a fully-drained stream releases its slot."""
    resp = _FakeResp()
    router, client = _bounded_router(monkeypatch, resp)

    ret = await router.stream_request(
        "http://127.0.0.1:9", "m", {"messages": []},
        backend_key="k", slot_wait_timeout=1.0,
    )
    body_iter = ret[1]
    assert client.inflight_used == 1

    await _drain(body_iter)

    assert client.inflight_used == 0


# ---------------------------------------------------------------------------
# Background-task wiring: the slot/connection cleanup must be attached to the
# StreamingResponse so it runs when FastAPI closes the response — including the
# client-disconnect path, where the response generator may only be finalized by
# GC. On this stack (uvicorn ASGI spec 2.3) Starlette runs response.background
# after the task group exits, disconnect included.
# ---------------------------------------------------------------------------

def _make_minimal_config(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True},
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            "test-model": {"provider": "local-llm", "class": "unknown", "privacy": "local_only"},
        }
    }))
    return cfg_dir


async def test_streaming_response_attaches_cleanup_as_background(tmp_path, monkeypatch):
    """The chat-completions streaming path must hand the per-request cleanup to
    ``StreamingResponse(background=...)``. This pins the wiring (does the right
    callable get attached?) independently of whether a disconnect ever fires —
    a unit test of cleanup() alone would not catch app.py forgetting to wire it.
    """
    app = create_app(config_dir=_make_minimal_config(tmp_path))

    async def _fake_cleanup() -> None:  # identity sentinel
        return None

    async def _fake_body_iter():
        yield b"data: [DONE]\n\n"

    async def _fake_route_and_forward(request_data, headers=None, stream=False):
        result = RouteResult(
            success=True, selected_model="test-model", backend_url="http://127.0.0.1",
            provider_name="local-llm", decision={"ranked": ["test-model"]},
        )
        upstream = httpx.Response(200, headers={"content-type": "text/event-stream"})
        return upstream, _fake_body_iter(), result, _fake_cleanup

    monkeypatch.setattr(app.state.router, "route_and_forward", _fake_route_and_forward)

    captured: dict = {}

    def _spy_streaming_response(content, **kwargs):
        captured["background"] = kwargs.get("background")
        return RealStreamingResponse(content, **kwargs)

    monkeypatch.setattr(app_mod, "StreamingResponse", _spy_streaming_response)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200

    bg = captured.get("background")
    assert bg is not None, "streaming response was built without a background cleanup task"
    assert bg.func is _fake_cleanup, "background task must wrap the per-request cleanup callable"


# ---------------------------------------------------------------------------
# Fix #5: churn integration. Deterministic stand-in for "kill the upstream
# mid-flight (SIGKILL), restart it, verify routing recovers with no leaked
# counters." A real subprocess upstream is flaky in CI and the repo forbids real
# hosts, so we drive the same seams the daemon hits: an abrupt mid-stream
# connection drop (Fix #1 synchronous release) and a successful poll after an
# outage (Fix #3 backend-wipe). Recovery is bounded by one poll interval.
# ---------------------------------------------------------------------------

class _DyingResp:
    """Upstream that delivers one chunk then drops the connection mid-stream."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    async def aiter_raw(self):
        yield b"data: partial\n\n"
        raise httpx.RemoteProtocolError("peer closed connection")

    async def aclose(self):
        return None


def _dying_client_factory(*args, **kwargs):
    class _C:
        def build_request(self, *a, **k):
            return object()

        async def send(self, req, stream=False):
            return _DyingResp()

        async def aclose(self):
            return None

    return _C()


def _patch_models_ok(monkeypatch, model_ids):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"object": "list", "data": [{"id": m} for m in model_ids]}

    class _OkClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr(endpoint_mod.httpx, "AsyncClient", lambda *a, **k: _OkClient())


async def test_churn_midflight_kills_dont_leak_and_restart_recovers(monkeypatch):
    """A stream of requests whose upstream dies mid-flight must never leak slots;
    and after the backend restarts, full capacity is restored with clean
    counters — even for slots that genuinely leaked before recovery."""
    disc = DiscoveryManager()
    client = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1:9",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["m"]),
        circuit_breaker=CircuitBreaker(failure_threshold=2, recovery_timeout=0),
        max_concurrent=2,
    )
    disc.clients["k"] = client
    disc.model_to_client["m"] = "k"
    router = RequestRouter(config=None, discovery=disc)  # type: ignore[arg-type]

    monkeypatch.setattr(router_mod.httpx, "AsyncClient", _dying_client_factory)

    # Phase 1 — a stream of requests, each killed mid-flight. None may leak.
    for _ in range(5):
        ret = await router.stream_request(
            "http://127.0.0.1:9", "m", {"messages": []}, backend_key="k", slot_wait_timeout=1.0,
        )
        with contextlib.suppress(httpx.RemoteProtocolError):
            async for _chunk in ret[1]:
                pass
    assert client.inflight_used == 0, "mid-flight kills must not leak slots under churn"
    # Capacity fully intact: both slots still acquirable.
    h1 = await disc.acquire_slot_handle("k", 1.0)
    h2 = await disc.acquire_slot_handle("k", 1.0)
    assert client.inflight_used == 2
    h1.release()
    h2.release()
    assert client.inflight_used == 0

    # Phase 2 — two slots that genuinely leak (acquired, never released: both the
    # counter and the semaphore are stranded), and the backend goes down hard.
    _leaked1 = await disc.acquire_slot_handle("k", 1.0)
    _leaked2 = await disc.acquire_slot_handle("k", 1.0)
    assert client.inflight_used == 2
    with pytest.raises(SaturationError):  # capacity exhausted by the leak
        await disc.acquire_slot_handle("k", 0.05)
    client.state.circuit_open = True
    client.state.circuit_opened_at = time.monotonic() - 9999
    del _leaked1, _leaked2  # drop refs; release() is manual, so the slots stay leaked

    # Phase 3 — backend restarts: the next successful poll wipes stale state.
    _patch_models_ok(monkeypatch, ["m"])
    models = await client.fetch_models()

    assert models == ["m"]
    assert client.inflight_used == 0, "backend restart must wipe leaked in-flight slots"
    assert client.backend_resets == 1
    assert client.state.circuit_open is False
    # Routing recovers: capacity acquirable again (the leak no longer saturates).
    h = await disc.acquire_slot_handle("k", 1.0)
    assert client.inflight_used == 1
    h.release()


async def test_streaming_request_end_to_end_releases_slot(tmp_path, monkeypatch):
    """Real composition with no fake at the slot boundary: a streaming request
    flows app.py → route_and_forward → stream_request against a real bounded
    backend, the real slot is acquired, and after the response completes
    inflight_used is back to 0. Catches a broken 4-tuple unpack or unwired
    cleanup that the per-boundary tests (each faking one side) cannot.

    Happy path: the slot is freed by the iterator's finally on full drain (the
    background task's disconnect role is covered by the cancelled-aclose unit
    test — a mid-stream disconnect resists ASGITransport).
    """
    app = create_app(config_dir=_make_minimal_config(tmp_path))
    disc = app.state.discovery
    key = compose_backend_key("local-llm", None, "")
    client = EndpointClient(
        provider_name="local-llm", base_url="http://127.0.0.1",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["test-model"]),
        circuit_breaker=CircuitBreaker(), max_concurrent=2,
    )
    disc.clients[key] = client
    disc.model_to_client["test-model"] = key

    class _SSEResp:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            yield b"data: {}\n\n"
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            return None

    # httpx.AsyncClient is module-global, so the patch also intercepts the test's
    # own ASGI client. Delegate to the real client when a transport is passed
    # (the test harness); return the fake only for the router's upstream call.
    real_async_client = httpx.AsyncClient

    def _sse_factory(*a, **k):
        if "transport" in k:
            return real_async_client(*a, **k)

        class _C:
            def build_request(self, *a, **k):
                return object()

            async def send(self, req, stream=False):
                return _SSEResp()

            async def aclose(self):
                return None

        return _C()

    monkeypatch.setattr(router_mod.httpx, "AsyncClient", _sse_factory)

    # Prove the real slot path actually ran (not a no-op handle from a key miss).
    acquired_keys: list[str] = []
    orig = disc.acquire_slot_handle

    async def _spy(k, wait_timeout):
        acquired_keys.append(k)
        return await orig(k, wait_timeout)

    monkeypatch.setattr(disc, "acquire_slot_handle", _spy)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        resp = await http.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        body = resp.content  # fully consume the stream

    assert b"DONE" in body
    assert acquired_keys == [key], "the real bounded backend's slot must have been acquired"
    assert client.inflight_used == 0, "slot must be released after the streaming response completes"


# ---------------------------------------------------------------------------
# F3: streaming outcome fidelity. The recorded `outcome` must reflect how the
# stream actually terminated, not just the initial upstream status — so a
# 200-then-stall stops being logged as success. (The client-disconnect branch
# is covered by the pure classifier unit tests; a real disconnect resists
# ASGITransport, see above.)
# ---------------------------------------------------------------------------

def _stream_app_emitting(tmp_path, monkeypatch, body_chunks, status=200):
    """App whose streaming route yields ``body_chunks``; returns a dict that
    captures the kwargs of the single emit_chat_completion call."""
    app = create_app(config_dir=_make_minimal_config(tmp_path))

    async def _cleanup() -> None:
        return None

    async def _body():
        for c in body_chunks:
            yield c

    async def _rf(request_data, headers=None, stream=False):
        result = RouteResult(
            success=True, selected_model="test-model", backend_url="http://127.0.0.1",
            provider_name="local-llm", decision={"ranked": ["test-model"]},
        )
        upstream = httpx.Response(status, headers={"content-type": "text/event-stream"})
        return upstream, _body(), result, _cleanup

    monkeypatch.setattr(app.state.router, "route_and_forward", _rf)

    captured: dict = {}

    def _spy_emit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(app_mod, "emit_chat_completion", _spy_emit)
    return app, captured


async def test_streaming_outcome_success_on_clean_done(tmp_path, monkeypatch):
    """A stream that ends with [DONE] records outcome='success'."""
    app, captured = _stream_app_emitting(
        tmp_path, monkeypatch,
        [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"],
    )
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        resp = await http.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        _ = resp.content  # drain
    assert captured.get("outcome") == "success"


async def test_streaming_outcome_incomplete_when_no_terminal_marker(tmp_path, monkeypatch):
    """A 200 stream that ends WITHOUT [DONE]/finish_reason records
    'stream_incomplete', not 'success' — the silent-hangup fix."""
    app, captured = _stream_app_emitting(
        tmp_path, monkeypatch,
        [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'],
    )
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        resp = await http.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        _ = resp.content  # drain
    assert captured.get("outcome") == "stream_incomplete"


async def test_streaming_captures_ttft_when_chunks_flow(tmp_path, monkeypatch):
    """A streaming response records a non-None time-to-first-token (ns)."""
    app, captured = _stream_app_emitting(
        tmp_path, monkeypatch,
        [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"],
    )
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        resp = await http.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        _ = resp.content  # drain
    assert captured.get("ttft_ns") is not None
    assert captured["ttft_ns"] >= 0


async def test_streaming_ttft_none_when_no_chunks(tmp_path, monkeypatch):
    """An empty stream (zero chunks) records ttft_ns=None — no first token to time."""
    app, captured = _stream_app_emitting(tmp_path, monkeypatch, [])
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        resp = await http.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 200
        _ = resp.content  # drain
    assert captured.get("ttft_ns") is None
