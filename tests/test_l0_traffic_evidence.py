"""L0 discovery must not mark a backend unavailable while real completions
are flowing.

The 5s /v1/models poll shares one saturated pipe with multi-MB long-context
uploads, so it starves exactly when the backend is busiest. Before this guard,
ONE starved poll set status=unavailable and wiped the model list — every
named-model request 503'd for a backend that was returning 200s at that very
moment (gb200, evening of 2026-08-21/22) — and three starved polls opened the
silent L0 breaker, stretching the outage to recovery_timeout. Fresh completed
traffic (endpoint.traffic_is_fresh, stamped by the router) vetoes both the
poll verdict and the breaker count, mirroring the L2 probe's guard. A
genuinely dead backend completes nothing, loses its evidence within
TRAFFIC_FRESH_S, and flips unavailable as before.
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_relay.config.types import EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient, TRAFFIC_FRESH_S, traffic_is_fresh
from llm_relay.discovery.manager import DiscoveryManager


def _client(models=("glm-5.2-nvfp4",), status=EndpointStatus.healthy):
    c = EndpointClient(provider_name="gb200", base_url="http://127.0.0.1:9")
    c.state.status = status
    c.state.models = list(models)
    return c


def _failing_httpx():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("poll starved"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def test_traffic_is_fresh_window():
    c = _client()
    assert not traffic_is_fresh(c)  # no evidence yet
    c.last_traffic_success_ns = time.time_ns()
    assert traffic_is_fresh(c)
    c.last_traffic_success_ns = time.time_ns() - int((TRAFFIC_FRESH_S + 5) * 1e9)
    assert not traffic_is_fresh(c)


async def test_fetch_models_failure_not_counted_when_traffic_fresh():
    """A starved poll with fresh traffic must not advance the L0 breaker."""
    c = _client()
    c.last_traffic_success_ns = time.time_ns()
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _failing_httpx()
        for _ in range(c.circuit_breaker.failure_threshold + 2):
            assert await c.fetch_models() == []
    assert c.state.consecutive_failures == 0
    assert not c.state.circuit_open


async def test_fetch_models_failure_counts_when_traffic_stale():
    """With no (or stale) traffic evidence the breaker behaves as before."""
    c = _client()
    c.last_traffic_success_ns = time.time_ns() - int((TRAFFIC_FRESH_S + 60) * 1e9)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _failing_httpx()
        for _ in range(c.circuit_breaker.failure_threshold):
            await c.fetch_models()
    assert c.state.circuit_open


async def test_poll_keeps_catalog_when_traffic_fresh():
    """An empty poll result while completions are flowing keeps the previous
    status and model list — stale-but-live beats absent."""
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns()
    c.fetch_models = AsyncMock(return_value=[])
    await disc._poll_client_once(c)
    assert c.state.status == EndpointStatus.healthy
    assert c.state.models == ["glm-5.2-nvfp4"]


async def test_poll_marks_unavailable_when_traffic_stale():
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns() - int((TRAFFIC_FRESH_S + 60) * 1e9)
    c.fetch_models = AsyncMock(return_value=[])
    await disc._poll_client_once(c)
    assert c.state.status == EndpointStatus.unavailable
    assert c.state.models == []


async def test_poll_marks_unavailable_when_no_prior_models():
    """Traffic evidence never resurrects a backend that had nothing loaded:
    with no prior catalog there is nothing to preserve."""
    disc = DiscoveryManager()
    c = _client(models=(), status=EndpointStatus.unavailable)
    c.last_traffic_success_ns = time.time_ns()
    c.fetch_models = AsyncMock(return_value=[])
    await disc._poll_client_once(c)
    assert c.state.status == EndpointStatus.unavailable


async def test_poll_recovery_still_promotes():
    """A successful poll still promotes unavailable -> healthy and refreshes
    the catalog (unchanged behavior)."""
    disc = DiscoveryManager()
    c = _client(models=(), status=EndpointStatus.unavailable)
    c.fetch_models = AsyncMock(return_value=["glm-5.2-nvfp4"])
    await disc._poll_client_once(c)
    assert c.state.status == EndpointStatus.healthy
    assert c.state.models == ["glm-5.2-nvfp4"]
