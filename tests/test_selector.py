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


def test_get_model_state_unavailable_when_backend_healthy_but_not_serving_model():
    """Availability is per-MODEL, not per-backend.

    A model mapped to a healthy backend that is NOT actually reporting it (e.g. the
    box was reimaged and relaunched with a different served-model-name) must read
    unavailable. Otherwise the router selects it and the upstream returns 404 — the
    dominant production failure mode. The backend being up is not enough; it must be
    serving THIS model.
    """
    disc = DiscoveryManager()
    # Backend is healthy, but its reported model set does NOT include our model.
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy, models=["a-different-model"]
    )
    disc.clients["k1"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["qwen3.5-9b"] = "k1"

    assert disc.get_model_state("qwen3.5-9b") == ModelStatus.unavailable


def test_get_model_state_available_when_backend_serves_model():
    """Regression guard for the happy path: a healthy backend that DOES report the
    model reads available."""
    disc = DiscoveryManager()
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-9b"]
    )
    disc.clients["k1"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["qwen3.5-9b"] = "k1"

    assert disc.get_model_state("qwen3.5-9b") == ModelStatus.available


def test_get_model_state_matches_served_model_name_override():
    """Explicit served_model_name: a backend reporting a different served id is
    available when the config declares that served name."""
    disc = DiscoveryManager()
    disc.served_names["model-x"] = "Model-X-7B-UD-Q4_K_XL.gguf"
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy,
        models=["Model-X-7B-UD-Q4_K_XL.gguf"],
    )
    disc.clients["k1"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["model-x"] = "k1"

    assert disc.get_model_state("model-x") == ModelStatus.available


def test_get_model_state_fuzzy_matches_gguf_filename_without_override():
    """Robustness: even with no explicit override, a backend reporting a GGUF
    filename that contains the config name (case-insensitive) is recognized as
    serving it — so llama.cpp backends reporting '<Name>-UD-Q4_K_XL.gguf' don't
    read unavailable. Mirrors the /status _actually_available convention."""
    disc = DiscoveryManager()
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy,
        models=["Model-X-7B-UD-Q4_K_XL.gguf"],
    )
    disc.clients["k1"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["model-x"] = "k1"

    assert disc.get_model_state("model-x") == ModelStatus.available


def test_get_model_state_unavailable_when_serving_an_unrelated_model():
    """Issue-2 guard preserved: a healthy backend serving an UNRELATED model (config
    name neither present nor a substring) must still read unavailable."""
    disc = DiscoveryManager()
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy, models=["model-y"],
    )
    disc.clients["k1"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["model-x"] = "k1"

    assert disc.get_model_state("model-x") == ModelStatus.unavailable


def test_loader_parses_served_model_name(tmp_path):
    """models.yaml `served_model_name` is parsed onto ModelConfig."""
    (tmp_path / "models.yaml").write_text(
        "models:\n  m1:\n    provider: p\n    served_model_name: M1-UD-Q4_K_XL.gguf\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    assert c.models.models["m1"].served_model_name == "M1-UD-Q4_K_XL.gguf"


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


# ---------------------------------------------------------------------------
# Context-window filter: must use LIVE max_model_len (matches advertisement),
# falling back to static config only when no backend reports a live value.
# ---------------------------------------------------------------------------

def _disc_with_live_ctx(model: str, live_ctx: int | None):
    """One healthy backend serving `model`; optionally seed its live
    max_model_len so get_live_context_window(model) returns `live_ctx`."""
    from llm_relay.routing.keys import compose_backend_key
    disc = DiscoveryManager()
    state = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=[model])
    if live_ctx is not None:
        state.model_max_lens = {model: live_ctx}
    key = compose_backend_key("local-llm", 8081, "")
    disc.clients[key] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client[model] = key
    return disc


def test_min_context_filter_uses_live_context_over_config():
    """When a backend's LIVE max_model_len is below the request floor, the
    candidate must be filtered out even though its static config context_window
    is large. Routing filters on live so it never admits a request the backend
    advertised it cannot hold (the relay's /v1/models reports live)."""
    c = _load()  # qwen3.5-35b static context_window is 262144
    disc = _disc_with_live_ctx("qwen3.5-35b", live_ctx=8192)
    sel = ModelSelector(c, disc)
    ctx = RoutingContext(requested_model="qwen3.5-35b", min_context=50000)
    assert sel._apply_constraints(ctx, ["qwen3.5-35b"]) == [], \
        "live 8192 < min_context 50000 must filter it, despite static config 262144"


def test_min_context_filter_falls_back_to_config_when_no_live():
    """No backend reports a live max_model_len -> the filter must fall back to
    the static config context_window (preserving the prior behavior)."""
    c = _load()  # qwen3.5-35b static context_window is 262144
    disc = _disc_with_live_ctx("qwen3.5-35b", live_ctx=None)
    sel = ModelSelector(c, disc)
    # min_context below config (262144) -> kept; above config -> filtered.
    kept = sel._apply_constraints(RoutingContext(requested_model="qwen3.5-35b", min_context=50000), ["qwen3.5-35b"])
    assert kept == ["qwen3.5-35b"], "no live value -> config 262144 >= 50000 keeps it"
    filtered = sel._apply_constraints(RoutingContext(requested_model="qwen3.5-35b", min_context=300000), ["qwen3.5-35b"])
    assert filtered == [], "no live value -> config 262144 < 300000 filters it"


def test_strict_single_candidate_skips_load_sort(monkeypatch):
    """In strict mode with the requested model present there is exactly one
    candidate — there is no load decision to make, so _sort_by_load must be
    skipped entirely. Defensive: it keeps a corrupt inflight counter on the one
    backend from ever touching the routing decision."""
    c = _load()
    c.policy.explicit.strict = True
    # 9b carries (possibly corrupt) load; in strict mode it must not matter.
    disc = _two_backend_disc(load_9b=2, load_35b=0)
    sel = ModelSelector(c, disc)

    calls: list[list[str]] = []
    orig = sel._sort_by_load
    monkeypatch.setattr(sel, "_sort_by_load", lambda ranked: calls.append(list(ranked)) or orig(ranked))

    ctx = RoutingContext(requested_model="qwen3.5-9b")
    ranked = sel._prepare_ranked(ctx)

    assert ranked == ["qwen3.5-9b"]
    assert calls == [], "single ordered candidate (strict mode) must not invoke _sort_by_load"


# ---------------------------------------------------------------------------
# Host-qualified model identity: a 'provider:model' request resolves to the
# bare concrete model (and validates the provider), so the same model on
# different hosts is individually addressable.
# ---------------------------------------------------------------------------

def test_qualified_id_hits_concrete_branch():
    """`provider:model` resolves to the bare concrete model (ordered branch),
    not the unordered discovery-availability fallback."""
    c = _load()  # qwen3.5-9b is served by provider 'local-llm' in the test config
    sel = ModelSelector(c, DiscoveryManager())
    candidates, ordered = sel._build_candidates(
        RoutingContext(requested_model="local-llm:qwen3.5-9b")
    )
    assert ordered is True, "qualified id must hit the concrete-model branch, not discovery fallback"
    assert candidates[0] == "qwen3.5-9b"


def test_qualified_id_mismatched_provider_does_not_resolve():
    """A provider that doesn't serve the model is not a valid pairing -> it must
    NOT resolve to the concrete model."""
    c = _load()
    sel = ModelSelector(c, DiscoveryManager())
    candidates, ordered = sel._build_candidates(
        RoutingContext(requested_model="wrong-prov:qwen3.5-9b")
    )
    assert ordered is False and candidates == [], "mismatched provider must not resolve to a concrete model"
