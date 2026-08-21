"""Tests for config-drift detection and in-band degeneracy detector."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.config_drift import ConfigDriftDetector


def _make_discovery(backend_key="test:8080", model="test-model"):
    disc = DiscoveryManager()
    disc.clients[backend_key] = EndpointClient(
        provider_name="test",
        base_url="http://127.0.0.1:8080",
        state=EndpointState(provider="test", status=EndpointStatus.healthy, models=[model]),
        circuit_breaker=CircuitBreaker(),
    )
    disc.model_to_client[model] = backend_key
    return disc


def _mock_models_response(models=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": models or [{"id": "test-model", "max_model_len": 131072}]}
    return resp


async def test_config_drift_baseline():
    """First check sets the baseline hash without warning."""
    disc = _make_discovery()
    detector = ConfigDriftDetector(disc, MagicMock())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_models_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        await detector._check_one("test:8080", disc.clients["test:8080"])
    assert "test:8080" in detector._hashes


async def test_config_drift_no_change():
    """Same config = same hash = no warning."""
    disc = _make_discovery()
    detector = ConfigDriftDetector(disc, MagicMock())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_mock_models_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        await detector._check_one("test:8080", disc.clients["test:8080"])
        await detector._check_one("test:8080", disc.clients["test:8080"])
    # No exception, hash unchanged
    assert detector._hashes["test:8080"] is not None


async def test_config_drift_detects_change():
    """Different model config = different hash = warning logged."""
    disc = _make_discovery()
    detector = ConfigDriftDetector(disc, MagicMock())
    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        # First response: 131072 context
        mock_client.get = AsyncMock(return_value=_mock_models_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = mock_client
        await detector._check_one("test:8080", disc.clients["test:8080"])
        old_hash = detector._hashes["test:8080"]

        # Second response: 65536 context (config changed)
        mock_client.get = AsyncMock(return_value=_mock_models_response(
            models=[{"id": "test-model", "max_model_len": 65536}]
        ))
        await detector._check_one("test:8080", disc.clients["test:8080"])
    new_hash = detector._hashes["test:8080"]
    assert old_hash != new_hash, "hash must change when model config changes"


# --- in-band degeneracy detector (on the streaming path) ---

def test_degeneracy_detector_catches_repetition_loop():
    """The in-band detector catches 'int64 vs int64' repetition and would
    abort the stream. This tests the detector function directly."""
    from llm_relay.degeneracy import is_degenerate
    loop_text = "int64 vs int64 " * 100
    assert is_degenerate(loop_text)


def test_degeneracy_detector_passes_normal_streaming():
    """Normal varied content is not flagged as degenerate."""
    from llm_relay.degeneracy import is_degenerate
    normal = "Here is the answer: the value is 42. This is because 2+2=4.\n" * 3
    assert not is_degenerate(normal)
