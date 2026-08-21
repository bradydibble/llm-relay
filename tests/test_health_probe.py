"""Tests for the L2 inference health probe (circuit breaker for wedged backends)."""
import asyncio
import json
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


def _mock_response(status_code=200, finish_reason="stop", completion_tokens=5):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": "OK"}}],
        "usage": {"completion_tokens": completion_tokens},
    }
    return resp


async def test_l2_probe_success_keeps_backend_healthy():
    """A healthy backend that responds to the L2 probe stays healthy."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_response())
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
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
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
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        await probe._probe_one("test:8080", disc.clients["test:8080"])
        await probe._probe_one("test:8080", disc.clients["test:8080"])
    assert disc.clients["test:8080"].state.status == EndpointStatus.degraded


async def test_l2_probe_abnormal_termination_counts_as_failure():
    """A response with 0 completion tokens (model didn't generate) is a failure."""
    disc = _make_discovery()
    probe = L2HealthProbe(disc, _make_config())
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_response(finish_reason="length", completion_tokens=0))
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
    disc.clients[key].state.status = EndpointStatus.degraded

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_response())
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
