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


# ---------------------------------------------------------------------------
# Load-aware selection: prefer least-loaded among equally-priority candidates
# ---------------------------------------------------------------------------

def _two_backend_disc(load_9b: int, load_35b: int, cap: int = 3):
    """Build a DiscoveryManager with qwen3.5-9b on `k9` and qwen3.5-35b on `k35`,
    both healthy, with the given inflight_used counters and a shared capacity."""
    disc = DiscoveryManager()
    s9 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-9b"])
    s35 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-35b"])
    c9 = EndpointClient(provider_name="local-llm", base_url="x", state=s9, max_concurrent=cap)
    c35 = EndpointClient(provider_name="local-llm", base_url="x", state=s35, max_concurrent=cap)
    c9.inflight_used = load_9b
    c35.inflight_used = load_35b
    # selector looks the client up via compose_backend_key(provider, port, path).
    # In tests we register under the same keys the loader would generate.
    from llm_relay.routing.keys import compose_backend_key
    k9 = compose_backend_key("local-llm", 8080, "")
    k35 = compose_backend_key("local-llm", 8081, "")
    disc.clients[k9] = c9
    disc.clients[k35] = c35
    disc.model_to_client["qwen3.5-9b"] = k9
    disc.model_to_client["qwen3.5-35b"] = k35
    return disc


def test_load_balance_idle_priority_wins_when_loads_tied():
    """When all candidates are idle (load=0), original alias priority decides.

    `subagent` chain is [qwen3.5-9b, qwen3.5-35b]. Both idle → 9b wins (priority).
    """
    c = _load()
    disc = _two_backend_disc(load_9b=0, load_35b=0)
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-9b"


def test_load_balance_overflows_when_priority_backend_loaded():
    """When the priority candidate is loaded and a later one is idle, the idle
    candidate wins — TTFT improves by skipping the slot wait.

    9b at 3/3 (100%), 35b at 0/3 (0%) → pick 35b despite 9b's higher priority.
    """
    c = _load()
    disc = _two_backend_disc(load_9b=3, load_35b=0)
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-35b", "saturated higher-priority backend should yield to idle alternate"


def test_load_balance_prefers_priority_when_neither_saturated():
    """A small load difference shouldn't flip the choice if neither is saturated:
    1/3 (33%) on the priority backend still beats 0/3 (0%) on the alternate IF
    we're comparing strict load ratio? No — we sort by (load_ratio, priority_idx).
    With 1/3 vs 0/3, ratios differ (0.333 > 0.0) so the alternate wins.

    This documents the chosen policy: ANY load on the priority backend yields
    to a fully-idle alternate. Brady's goal is TTFT; even one in-flight request
    can add multi-second slot wait. Aggressive overflow is the point.
    """
    c = _load()
    disc = _two_backend_disc(load_9b=1, load_35b=0)
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-35b"


def test_load_balance_picks_least_loaded_when_all_busy():
    """If every candidate has in-flight work, pick the least-loaded.

    9b at 3/3 (100%), 35b at 1/3 (33%) → pick 35b.
    """
    c = _load()
    disc = _two_backend_disc(load_9b=3, load_35b=1)
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-35b"


def test_load_balance_breaks_tie_on_original_priority():
    """When load ratios are equal, fall back to original alias priority.

    Both at 1/3 (33%) → 9b wins because it's first in the subagent alias.
    """
    c = _load()
    disc = _two_backend_disc(load_9b=1, load_35b=1)
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-9b"


def test_load_balance_treats_no_semaphore_as_idle():
    """Backends without max_concurrent (unbounded, no inflight tracking) score
    as load=0.0 — equivalent to fully-idle.

    Priority backend has max_concurrent=None, alternate has 2/3 in flight.
    Priority backend's load is 0.0; alternate's is 0.67. Priority wins.
    """
    c = _load()
    disc = _two_backend_disc(load_9b=0, load_35b=2, cap=3)
    # Override: make 9b unbounded (max_concurrent=None, no inflight_sem).
    from llm_relay.routing.keys import compose_backend_key
    k9 = compose_backend_key("local-llm", 8080, "")
    disc.clients[k9].max_concurrent = None
    disc.clients[k9].inflight_sem = None
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-9b"


def test_load_balance_select_chain_orders_by_load():
    """select_chain (used by route_and_forward retry) must respect the same
    load ordering — first candidate is least-loaded so retries fall back to
    progressively more contended backends."""
    c = _load()
    disc = _two_backend_disc(load_9b=3, load_35b=0)
    sel = ModelSelector(c, disc)
    chain = sel.select_chain(RoutingContext(requested_model="subagent"))
    assert [cand.model for cand in chain] == ["qwen3.5-35b", "qwen3.5-9b"]
