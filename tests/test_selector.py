"""Selector and discovery tests."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import CircuitBreaker, EndpointState, EndpointStatus, ModelStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.selector import ModelSelector, RoutingContext


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load() -> ConfigLoader:
    c = ConfigLoader(config_dir=CONFIG_DIR)
    c.load()
    return c


def test_aliases_loaded():
    c = _load()
    assert "subagent" in c.models.aliases
    assert c.models.aliases["subagent"][0] == "qwen3.5-9b"


def test_models_loaded_with_ports():
    c = _load()
    assert c.models.models["qwen3.5-9b"].port == 8080
    assert c.models.models["qwen3.5-35b"].port == 8081


def test_alias_order_is_preserved_in_selection():
    """The first available candidate in alias order should win, even when a
    later candidate has a higher preference."""
    c = _load()
    disc = DiscoveryManager()

    # Mark BOTH 9B and 35B as available, with 35B having higher preference (0.9 > 0.7).
    state_9 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-9b"])
    state_35 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-35b"])
    disc.clients["k9"] = EndpointClient(provider_name="local-llm", base_url="x", state=state_9)
    disc.clients["k35"] = EndpointClient(provider_name="local-llm", base_url="x", state=state_35)
    disc.model_to_client["qwen3.5-9b"] = "k9"
    disc.model_to_client["qwen3.5-35b"] = "k35"

    sel = ModelSelector(c, disc)
    ctx = RoutingContext(requested_model="subagent")
    pick = sel.select_best(ctx)
    assert pick == "qwen3.5-9b", "alias order [9b, 35b] must beat preference rank"


def test_alias_skips_unavailable():
    c = _load()
    disc = DiscoveryManager()
    state_9 = EndpointState(provider="local-llm", status=EndpointStatus.unavailable, models=[])
    state_35 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-35b"])
    disc.clients["k9"] = EndpointClient(provider_name="local-llm", base_url="x", state=state_9)
    disc.clients["k35"] = EndpointClient(provider_name="local-llm", base_url="x", state=state_35)
    disc.model_to_client["qwen3.5-9b"] = "k9"
    disc.model_to_client["qwen3.5-35b"] = "k35"

    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-35b"


def test_privacy_local_only_excludes_cloud():
    c = _load()
    disc = DiscoveryManager()
    sel = ModelSelector(c, disc)
    ctx = RoutingContext(requested_model="claude-3-5-sonnet")
    # Default privacy is local_only — cloud-only model should not survive filter.
    pick = sel.select_best(ctx)
    assert pick is None


def test_circuit_breaker_recovers_after_timeout():
    """After recovery_timeout elapses, the breaker resets so polling can probe again."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    state = EndpointState(provider="test")
    client = EndpointClient(provider_name="test", base_url="http://nope", state=state, circuit_breaker=breaker)

    # Trip the breaker.
    client._record_failure()
    client._record_failure()
    assert client.state.circuit_open is True
    assert client.state.circuit_opened_at is not None

    # Pretend recovery_timeout has elapsed.
    client.state.circuit_opened_at = time.monotonic() - 5
    client._maybe_recover_circuit()
    assert client.state.circuit_open is False
    assert client.state.consecutive_failures == 0
