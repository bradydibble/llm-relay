"""Observation-first health (docs/observation-first-health-spec.md).

Covers the four mechanisms landed together:
  §3.1 symmetric traffic evidence — real-request failures take a backend down
       faster than probes; one real success outranks everything;
  §3.2 probe demotion — polls and L2 probes are skipped while observation
       already answers;
  §3.3 named-model live check — no refusal from cached state;
  §3.5 persisted priors — a restarted relay resumes the fleet picture.
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_relay.config.types import EndpointStatus
from llm_relay.discovery.endpoint import (
    TRAFFIC_FAILURES_TO_MARK_DOWN,
    TRAFFIC_FRESH_S,
    EndpointClient,
)
from llm_relay.discovery.manager import (
    PRIORS_MAX_AGE_S,
    STATE_FILE_VERSION,
    DiscoveryManager,
    _backend_state_path,
)
from llm_relay.health import L2HealthProbe
from llm_relay.routing.router import RequestRouter


def _client(models=("glm-5.2-nvfp4",), status=EndpointStatus.healthy):
    c = EndpointClient(provider_name="gb200", base_url="http://127.0.0.1:9")
    c.state.status = status
    c.state.models = list(models)
    return c


def _mock_httpx(response=None, exc=None):
    mock_client = AsyncMock()
    if exc is not None:
        mock_client.get = AsyncMock(side_effect=exc)
        mock_client.post = AsyncMock(side_effect=exc)
    else:
        mock_client.get = AsyncMock(return_value=response)
        mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# --- §3.1 symmetric evidence -------------------------------------------------

def test_traffic_success_promotes_and_heals():
    c = _client(status=EndpointStatus.unavailable)
    c.consecutive_traffic_failures = 2
    c.state.circuit_open = True
    c.state.circuit_opened_at = time.monotonic()
    c.note_traffic_success()
    assert c.state.status == EndpointStatus.healthy
    assert not c.state.circuit_open
    assert c.consecutive_traffic_failures == 0
    assert c.last_traffic_success_ns is not None


def test_traffic_failures_take_backend_down():
    c = _client()
    for _ in range(TRAFFIC_FAILURES_TO_MARK_DOWN - 1):
        c.note_traffic_failure()
    assert c.state.status == EndpointStatus.healthy  # not yet
    c.note_traffic_failure()
    assert c.state.status == EndpointStatus.unavailable
    assert c.state.circuit_open


def test_interleaved_success_resets_failure_streak():
    c = _client()
    c.note_traffic_failure()
    c.note_traffic_failure()
    c.note_traffic_success()
    c.note_traffic_failure()
    c.note_traffic_failure()
    assert c.state.status == EndpointStatus.healthy


async def test_router_stamps_transport_failure():
    disc = DiscoveryManager()
    c = _client()
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(exc=httpx.ConnectError("refused"))
        with pytest.raises(httpx.ConnectError):
            await router.forward_request(
                "http://127.0.0.1:9/v1", "glm-5.2-nvfp4", {"messages": []},
                backend_key="gb200:8000",
            )
    assert c.consecutive_traffic_failures == 1


async def test_router_stamps_backend_5xx_but_not_4xx():
    disc = DiscoveryManager()
    c = _client()
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    resp502 = MagicMock(); resp502.status_code = 502
    resp400 = MagicMock(); resp400.status_code = 400
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(response=resp502)
        await router.forward_request(
            "http://127.0.0.1:9/v1", "glm-5.2-nvfp4", {"messages": []},
            backend_key="gb200:8000",
        )
    assert c.consecutive_traffic_failures == 1
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(response=resp400)
        await router.forward_request(
            "http://127.0.0.1:9/v1", "glm-5.2-nvfp4", {"messages": []},
            backend_key="gb200:8000",
        )
    assert c.consecutive_traffic_failures == 1  # 4xx is never evidence


# --- §3.2 probe demotion ------------------------------------------------------

async def test_poll_skipped_entirely_under_fresh_traffic():
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns()
    c.fetch_models = AsyncMock()
    await disc._poll_client_once(c)
    c.fetch_models.assert_not_called()
    assert c.state.last_poll is not None


async def test_poll_still_runs_when_catalog_empty():
    """Fresh traffic never leaves a backend stuck catalogless: an empty model
    list always polls."""
    disc = DiscoveryManager()
    c = _client(models=())
    c.last_traffic_success_ns = time.time_ns()
    c.fetch_models = AsyncMock(return_value=["glm-5.2-nvfp4"])
    await disc._poll_client_once(c)
    c.fetch_models.assert_called_once()
    assert c.state.models == ["glm-5.2-nvfp4"]


async def test_l2_probe_skipped_under_fresh_traffic():
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns()
    disc.clients["gb200:8000"] = c
    probe = L2HealthProbe(disc, MagicMock())
    with patch.object(probe, "_probe_one", new_callable=AsyncMock) as mock_probe:
        await probe._probe_all()
    mock_probe.assert_not_called()


async def test_l2_probe_still_runs_when_idle():
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns() - int((TRAFFIC_FRESH_S + 30) * 1e9)
    disc.clients["gb200:8000"] = c
    probe = L2HealthProbe(disc, MagicMock())
    with patch.object(probe, "_probe_one", new_callable=AsyncMock) as mock_probe:
        await probe._probe_all()
    mock_probe.assert_called_once()


# --- §3.3 named-model live check ---------------------------------------------

def _models_response(ids):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": [{"id": i} for i in ids]})
    return resp


async def test_live_check_recovers_stale_unavailable():
    disc = DiscoveryManager()
    c = _client(models=(), status=EndpointStatus.unavailable)
    c.state.circuit_open = True
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(response=_models_response(["glm-5.2-nvfp4"]))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert recovered
    assert c.state.status == EndpointStatus.healthy
    assert c.state.models == ["glm-5.2-nvfp4"]
    assert not c.state.circuit_open
    assert avail["checked_live"] is True


async def test_live_check_refused_maps_to_starting_with_recent_traffic():
    disc = DiscoveryManager()
    c = _client(status=EndpointStatus.unavailable)
    c.last_traffic_success_ns = time.time_ns() - int(600 * 1e9)  # 10 min ago
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(exc=httpx.ConnectError("refused"))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert not recovered
    assert avail["reason"] == "starting"


async def test_live_check_refused_maps_to_refused_when_cold():
    disc = DiscoveryManager()
    c = _client(status=EndpointStatus.unavailable)
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(exc=httpx.ConnectError("refused"))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert not recovered
    assert avail["reason"] == "refused"


async def test_live_check_timeout_reason():
    disc = DiscoveryManager()
    c = _client(status=EndpointStatus.unavailable)
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(exc=httpx.ConnectTimeout("slow"))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert not recovered
    assert avail["reason"] == "timeout"


async def test_live_check_not_loaded():
    disc = DiscoveryManager()
    c = _client(status=EndpointStatus.unavailable)
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(response=_models_response(["other-model"]))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert not recovered
    assert avail["reason"] == "not_loaded"


async def test_live_check_never_overrules_wedge_verdict():
    """Listening and listing != generating: L2's degraded stands (invariant 5)."""
    disc = DiscoveryManager()
    c = _client(status=EndpointStatus.degraded)
    disc.clients["gb200:8000"] = c
    router = RequestRouter(None, disc)
    with patch("httpx.AsyncClient") as cls:
        cls.return_value = _mock_httpx(response=_models_response(["glm-5.2-nvfp4"]))
        recovered, avail = await router._named_live_check("glm-5.2-nvfp4", c)
    assert not recovered
    assert avail["reason"] == "degraded"
    assert c.state.status == EndpointStatus.degraded


# --- §3.5 persisted priors ----------------------------------------------------

def _write_state(tmp_path, backends, age_s=0.0):
    doc = {
        "version": STATE_FILE_VERSION,
        "saved_at_ns": time.time_ns() - int(age_s * 1e9),
        "backends": backends,
    }
    (tmp_path / "backend-state.json").write_text(json.dumps(doc))


async def _register(disc, key):
    await disc.register_backend(
        key=key, provider_name="gb200", base_url="http://127.0.0.1:9",
        models_hint=[], poll_interval=3600,
    )
    for t in disc._tasks:
        t.cancel()


async def test_fresh_priors_adopted(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_RELAY_STATE_DIR", str(tmp_path))
    _write_state(tmp_path, {
        "gb200:8000": {
            "status": "healthy",
            "models": ["glm-5.2-nvfp4"],
            "last_traffic_success_ns": time.time_ns() - int(30 * 1e9),
        },
    })
    disc = DiscoveryManager()
    await _register(disc, "gb200:8000")
    c = disc.clients["gb200:8000"]
    assert c.state.status == EndpointStatus.healthy
    assert c.state.models == ["glm-5.2-nvfp4"]
    assert c.last_traffic_success_ns is not None
    assert disc.model_to_client["glm-5.2-nvfp4"] == "gb200:8000"


async def test_stale_priors_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_RELAY_STATE_DIR", str(tmp_path))
    _write_state(tmp_path, {
        "gb200:8000": {"status": "healthy", "models": ["glm-5.2-nvfp4"],
                        "last_traffic_success_ns": time.time_ns()},
    }, age_s=PRIORS_MAX_AGE_S + 60)
    disc = DiscoveryManager()
    await _register(disc, "gb200:8000")
    c = disc.clients["gb200:8000"]
    assert c.state.models == []
    assert c.last_traffic_success_ns is None


async def test_corrupt_priors_never_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_RELAY_STATE_DIR", str(tmp_path))
    (tmp_path / "backend-state.json").write_text("{not json")
    disc = DiscoveryManager()
    await _register(disc, "gb200:8000")
    assert disc.clients["gb200:8000"].state.models == []


def test_snapshot_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_RELAY_STATE_DIR", str(tmp_path))
    disc = DiscoveryManager()
    c = _client()
    c.last_traffic_success_ns = time.time_ns()
    disc.clients["gb200:8000"] = c
    snap = disc._state_snapshot()
    (tmp_path / "backend-state.json").write_text(json.dumps(snap))
    fresh = DiscoveryManager()
    entry = fresh._load_priors().get("gb200:8000")
    assert entry["status"] == "healthy"
    assert entry["models"] == ["glm-5.2-nvfp4"]


def test_state_path_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_RELAY_STATE_DIR", raising=False)
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "relay" / "audit.log"))
    assert _backend_state_path() == tmp_path / "relay" / "backend-state.json"
    monkeypatch.delenv("LLM_RELAY_AUDIT_LOG", raising=False)
    assert _backend_state_path() is None
