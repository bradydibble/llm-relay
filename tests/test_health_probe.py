"""Tests for the L2 inference health probe (circuit breaker for wedged backends)."""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.health import L2HealthProbe, _BackendHealth, FAILURES_TO_OPEN, RECOVERY_PROBES


def _make_discovery(backend_key="test:8080", model="test-model", status=EndpointStatus.healthy):
    disc = DiscoveryManager()
    disc.clients[backend_key] = EndpointClient(
        provider_name="test",
        base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="test", status=status, models=[model]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client[model] = backend_key
    return disc


def _make_config():
    cfg = MagicMock()
    return cfg


def _mock_stream_response(status_code=200, has_data=True):
    """Mock a streaming SSE response for the L2 probe."""
    resp = MagicMock()
    resp.status_code = status_code
    # Simulate an async context manager for c.stream()
    # The probe reads the first chunk and checks for "data:" 
    async def _aiter_text():
        if has_data:
            yield 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
        yield 'data: [DONE]\n\n'
    resp.aiter_text = _aiter_text
    # The async context manager itself
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm


async def test_l2_probe_success_keeps_backend_healthy():
    """A healthy backend that responds to the L2 probe stays healthy."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_mock_stream_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        await probe._probe_one("test:8080", disc.clients["test:8080"])
    assert disc.clients["test:8080"].state.status == EndpointStatus.healthy


async def test_l2_probe_timeout_marks_degraded():
    """A backend that times out on the L2 probe is marked degraded after
    FAILURES_TO_OPEN consecutive failures. This is the wedged-slot detection."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        # First failure
        await probe._probe_one("test:8080", disc.clients["test:8080"])
        assert disc.clients["test:8080"].state.status == EndpointStatus.healthy  # not yet
        # Second failure — circuit opens
        await probe._probe_one("test:8080", disc.clients["test:8080"])
    assert disc.clients["test:8080"].state.status == EndpointStatus.degraded, \
        "backend must be degraded after 2 consecutive L2 probe failures"


async def test_l2_probe_error_counts_as_failure():
    """A connection error (not just timeout) also counts as a failure."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        await probe._probe_one("test:8080", disc.clients["test:8080"])
        await probe._probe_one("test:8080", disc.clients["test:8080"])
    assert disc.clients["test:8080"].state.status == EndpointStatus.degraded


async def test_l2_probe_no_sse_data_counts_as_failure():
    """A response with no SSE data (model didn't generate) is a failure."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_mock_stream_response(has_data=False))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        await probe._probe_one("test:8080", disc.clients["test:8080"])
        await probe._probe_one("test:8080", disc.clients["test:8080"])
    assert disc.clients["test:8080"].state.status == EndpointStatus.degraded


async def test_l2_circuit_recovery():
    """After the circuit opens, RECOVERY_PROBES consecutive successes close it
    and restore the backend to healthy."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    key = "test:8080"
    health = probe._health.setdefault(key, _BackendHealth())
    health.circuit_open = True
    health.consecutive_failures = FAILURES_TO_OPEN
    health.last_failure_ns = time.time_ns() - 120_000_000_000  # 120s ago (past cooldown)
    disc.clients[key].state.status = EndpointStatus.degraded

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_mock_stream_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        # First success (half-open)
        await probe._probe_one(key, disc.clients[key])
        assert disc.clients[key].state.status == EndpointStatus.degraded  # still degraded
        # Second success — circuit closes
        await probe._probe_one(key, disc.clients[key])
    assert disc.clients[key].state.status == EndpointStatus.healthy
    assert not probe._health[key].circuit_open


async def test_l2_skips_already_unavailable():
    """Don't probe backends that L0 already marked unavailable."""
    disc = _make_discovery(status=EndpointStatus.unavailable)
    probe = L2HealthProbe(disc, _make_config())
    # _probe_all should skip this backend
    with patch.object(probe, "_probe_one", new_callable=AsyncMock) as mock_probe:
        await probe._probe_all()
    mock_probe.assert_not_called()


async def test_l2_skips_backends_with_no_models():
    """Don't probe backends that have no models loaded."""
    disc = _make_discovery()
    disc.clients["test:8080"].state.models = []
    probe = L2HealthProbe(disc, _make_config())
    with patch.object(probe, "_probe_one", new_callable=AsyncMock) as mock_probe:
        await probe._probe_all()
    mock_probe.assert_not_called()


# --- traffic-evidence tests: probe starvation under long-context load must ---
# --- not open the circuit while real completions are flowing (2026-08-21). ---

def _failing_async_client():
    """httpx.AsyncClient mock whose stream() always times out."""
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(side_effect=httpx.ReadTimeout("timed out"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


async def test_l2_probe_timeout_not_counted_when_traffic_fresh():
    """A probe timeout while real requests are completing is probe starvation
    (a 512k prefill hogs the scheduler past the probe deadline), not a wedge.
    The circuit must stay closed however many probes starve."""
    disc = _make_discovery()
    ep = disc.clients["test:8080"]
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value = _failing_async_client()
        for _ in range(FAILURES_TO_OPEN + 2):
            ep.last_traffic_success_ns = time.time_ns()  # traffic keeps completing
            await probe._probe_one("test:8080", ep)
    assert ep.state.status == EndpointStatus.healthy
    assert not probe._health["test:8080"].circuit_open


async def test_l2_probe_timeout_counts_when_traffic_stale():
    """Traffic evidence expires: with the last real success far in the past,
    probe failures count and the circuit opens as before."""
    disc = _make_discovery()
    ep = disc.clients["test:8080"]
    ep.last_traffic_success_ns = time.time_ns() - 600_000_000_000  # 10 min ago
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value = _failing_async_client()
        await probe._probe_one("test:8080", ep)
        await probe._probe_one("test:8080", ep)
    assert ep.state.status == EndpointStatus.degraded
    assert probe._health["test:8080"].circuit_open


async def test_l2_fresh_traffic_resets_failure_streak():
    """A busy-skipped probe resets the failure streak, so single failures in
    separate busy windows never accumulate into a spurious open."""
    disc = _make_discovery()
    ep = disc.clients["test:8080"]
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value = _failing_async_client()
        await probe._probe_one("test:8080", ep)  # no traffic marker -> counted
        assert probe._health["test:8080"].consecutive_failures == 1
        ep.last_traffic_success_ns = time.time_ns()
        await probe._probe_one("test:8080", ep)  # busy -> skipped AND reset
        assert probe._health["test:8080"].consecutive_failures == 0
        ep.last_traffic_success_ns = None  # traffic stops
        await probe._probe_one("test:8080", ep)  # counted (1)
        assert ep.state.status == EndpointStatus.healthy
        await probe._probe_one("test:8080", ep)  # counted (2) -> opens
    assert ep.state.status == EndpointStatus.degraded


async def test_router_forward_success_stamps_traffic_evidence():
    """RequestRouter.forward_request records a completed 2xx on the endpoint,
    which is what the L2 probe consults as liveness evidence."""
    from llm_relay.routing.router import RequestRouter

    disc = _make_discovery()
    ep = disc.clients["test:8080"]
    router = RequestRouter(None, disc)
    resp = MagicMock()
    resp.status_code = 200
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        out = await router.forward_request(
            "http://127.0.0.1:8080/v1", "test-model", {"messages": []},
            backend_key="test:8080",
        )
    assert out is resp
    assert ep.last_traffic_success_ns is not None


async def test_router_forward_error_does_not_stamp():
    """A 5xx upstream response is not liveness evidence."""
    from llm_relay.routing.router import RequestRouter

    disc = _make_discovery()
    ep = disc.clients["test:8080"]
    router = RequestRouter(None, disc)
    resp = MagicMock()
    resp.status_code = 502
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        await router.forward_request(
            "http://127.0.0.1:8080/v1", "test-model", {"messages": []},
            backend_key="test:8080",
        )
    assert ep.last_traffic_success_ns is None
