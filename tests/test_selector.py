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


def test_public_config_aliases_match_canonical_lists():
    """Round-trip guard for the tag transpose: the loaded alias map (derived from
    per-model use_cases) must EXACTLY equal the canonical ordered lists, so a
    priority-encoding typo in the tags is caught."""
    c = _load()
    expected = {
        "main": ["qwen3.5-35b", "deepseek-r1-70b", "llama-3.3-70b", "qwen3.5-9b"],
        "subagent": ["qwen3.5-9b", "qwen3.5-35b"],
        "subagent-64k": ["qwen3.5-9b-64k", "qwen3.5-35b", "qwen3.5-9b"],
        "fast": ["qwen3.5-9b"],
        "high-quality": ["qwen3.5-35b", "deepseek-r1-70b", "llama-3.3-70b"],
        "long-context": ["qwen3.5-35b", "qwen3.5-9b-64k"],
        "reasoning": ["deepseek-r1-70b", "qwen3.5-35b"],
        "code_fast": ["qwen3.5-9b", "qwen3.5-35b"],
        "code_medium": ["qwen3.5-35b", "qwen3.5-9b"],
        "code_heavy": ["deepseek-r1-70b", "llama-3.3-70b", "qwen3.5-35b"],
    }
    assert c.models.aliases == expected


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
    monkeypatch.setattr(sel, "_sort_by_load", lambda ranked, *a, **k: calls.append(list(ranked)) or orig(ranked, *a, **k))

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


# ---------------------------------------------------------------------------
# Deterministic preference ranking (unknown-model branch). The weighted engine
# (quality/latency/cost/availability + per-request X-Llm-Relay-Weights) is gone;
# ordering is now preference desc, then name asc -- identical to MCP's
# select_for_capability, so the two discovery surfaces never disagree.
# ---------------------------------------------------------------------------

def test_rank_sorts_by_preference_then_name_only(tmp_path):
    """A non-local, higher-preference model must rank ahead of a local
    lower-preference one. Under the old weighted _rank the local model's
    latency/cost boost flipped this; pure preference ordering must not."""
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  alpha:\n    provider: p\n    preference: 0.3\n    tags: [local]\n"
        "  bravo:\n    provider: p\n    preference: 0.9\n    tags: []\n"
        "  zeta:\n    provider: p\n    preference: 0.9\n    tags: [local]\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    sel = ModelSelector(c, DiscoveryManager())
    # Input order scrambled; expect preference desc, ties broken by name asc.
    assert sel._rank(["zeta", "alpha", "bravo"]) == ["bravo", "zeta", "alpha"]


# ---------------------------------------------------------------------------
# Tag transpose: aliases (categories) are DERIVED from per-model use_cases tags
# at load — `aliases[uc] = models tagged uc, sorted by (uc-priority desc,
# preference desc, name asc)`. Model-major config: a model's whole story lives in
# one place. A static `aliases:` block is no longer supported — it is ignored at
# load with a warning; categories come only from tags.
# ---------------------------------------------------------------------------

def test_aliases_derived_from_model_use_cases(tmp_path):
    """A model's `use_cases: {uc: priority}` tags derive the alias map at load:
    higher uc-priority first, preference desc breaks a priority tie, name asc last."""
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  small:\n    provider: p\n    preference: 0.7\n    use_cases: {chat: 1, fast: 5}\n"
        "  big:\n    provider: p\n    preference: 0.9\n    use_cases: {chat: 5}\n"
        "  mid:\n    provider: p\n    preference: 0.8\n    use_cases: {chat: 5}\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    # chat: big & mid tie on priority 5 -> preference desc (0.9 > 0.8); small priority 1 last.
    assert c.models.aliases["chat"] == ["big", "mid", "small"]
    assert c.models.aliases["fast"] == ["small"]


def test_explicit_aliases_block_is_ignored_in_favor_of_tags(tmp_path, caplog):
    """Static `aliases:` blocks are sunset: a nested `aliases:` block is ignored
    at load (with a warning), and the category is derived purely from per-model
    `use_cases` tags. Here `aliases: {chat: [a]}` is dropped, so `chat` follows the
    tags — b (priority 5) before a (priority 1)."""
    import logging
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  a:\n    provider: p\n    use_cases: {chat: 1}\n"
        "  b:\n    provider: p\n    use_cases: {chat: 5}\n"
        "  aliases:\n    chat: [a]\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        c.load()
    assert c.models.aliases["chat"] == ["b", "a"], "explicit aliases block must be ignored; tags win"
    assert any("aliases" in r.getMessage().lower() for r in caplog.records), \
        "an ignored aliases block must warn so a legacy config can migrate"


def test_reasoning_floor_refuses_sub_floor_models(tmp_path):
    """A category with `reasoning_floor` (opt-in, off by default) only admits models
    whose preference clears the floor — the quality gate. A sub-floor model is
    refused even when it's the only thing live (better an honest no-candidate than
    quality below the bar); a model clearing the floor is selected normally."""
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  weak:\n    provider: local-llm\n    port: 8080\n    preference: 0.5\n    use_cases: {smart: 1}\n"
        "  strong:\n    provider: local-llm\n    port: 8081\n    preference: 0.95\n    use_cases: {smart: 2}\n"
        "  categories:\n    smart:\n      reasoning_floor: 0.9\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    # weak (0.5) is below the 0.9 floor -> excluded even though it's the only live model.
    sel = ModelSelector(c, _disc_serving("weak"))
    assert sel.select_best(RoutingContext(requested_model="smart")) is None, "sub-floor model must be refused"
    # strong (0.95) clears the floor -> selected.
    sel2 = ModelSelector(c, _disc_serving("strong"))
    assert sel2.select_best(RoutingContext(requested_model="smart")) == "strong"


def test_no_reasoning_floor_is_open_by_default(tmp_path):
    """Without a reasoning_floor, a category admits any live model in priority order
    — the floor is strictly opt-in."""
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  weak:\n    provider: local-llm\n    port: 8080\n    preference: 0.5\n    use_cases: {smart: 1}\n"
        "  strong:\n    provider: local-llm\n    port: 8081\n    preference: 0.95\n    use_cases: {smart: 2}\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    sel = ModelSelector(c, _disc_serving("weak"))  # only the weak model is live
    assert sel.select_best(RoutingContext(requested_model="smart")) == "weak", "no floor -> weak still serves"


# ---------------------------------------------------------------------------
# Open-by-default: an alias is a PRIORITY ORDER over the whole live fleet, not a
# whitelist. When its named members are unavailable, the request falls through to
# any other live model (preference-ranked) instead of dead-ending at the list.
# The named members stay the priority prefix; the live fleet is the tail. Context
# / privacy / reasoning-floor filters still run downstream, so the tail only ever
# yields a model that can actually serve.
# ---------------------------------------------------------------------------

def _disc_serving(*models: str):
    """A DiscoveryManager with one healthy backend per model name, each mapped and
    reporting only that model. Registered under arbitrary keys (not the composed
    backend key), so load-ratio lookups miss and score 0.0 (idle) — selection is
    then driven purely by priority/preference, not load."""
    disc = DiscoveryManager()
    for m in models:
        state = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=[m])
        disc.clients[f"k::{m}"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
        disc.model_to_client[m] = f"k::{m}"
    return disc


def test_alias_falls_through_to_live_model_outside_its_list():
    """subagent = [qwen3.5-9b, qwen3.5-35b]; both DOWN, but trinity-mini (NOT a
    member) is live -> the request must route to trinity-mini, not return None.
    This is the open-by-default principle: never dead-end while something's live."""
    c = _load()
    disc = _disc_serving("trinity-mini")  # neither subagent member is live
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "trinity-mini", "alias must fall through to the live fleet, not dead-end at its list"


def test_fleet_tail_includes_gguf_reported_backends():
    """The open-fallthrough tail must include heterogeneous (llama.cpp / GGUF)
    backends that report a filename id, not the config name. trinity-mini's backend
    reports 'Trinity-Mini-UD-Q4_K_XL.gguf' (fuzzy-matched via get_model_state);
    subagent's members are down -> the request must still fall through to it.
    Regression guard: enumerating raw backend-reported ids drops it (the GGUF id
    isn't a config key), silently excluding every GGUF backend from the tail."""
    c = _load()
    disc = DiscoveryManager()
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy,
        models=["Trinity-Mini-UD-Q4_K_XL.gguf"],  # GGUF filename, not the config name
    )
    disc.clients["k::trinity"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["trinity-mini"] = "k::trinity"
    sel = ModelSelector(c, disc)
    assert sel.select_best(RoutingContext(requested_model="subagent")) == "trinity-mini", \
        "fallthrough tail must include GGUF-reported backends (matched to config name via get_model_state)"


def test_fleet_tail_is_preference_ranked():
    """All named members down; multiple non-members live -> the higher-preference
    one wins the fallthrough. trinity-mini (pref 0.6) vs llama-3.3-70b (pref 0.9)
    -> llama wins."""
    c = _load()
    disc = _disc_serving("trinity-mini", "llama-3.3-70b")
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "llama-3.3-70b", "fallthrough tail must be ranked by preference desc"


def test_named_members_take_priority_over_fleet_tail_when_idle():
    """A live NAMED member beats a higher-preference non-member tail model when
    neither is under load: the named list is the priority prefix, the fleet is the
    tail. subagent's qwen3.5-9b (pref 0.7) live + deepseek-r1-70b (pref 0.95, not a
    member) live -> 9b still wins. Guards against ranking the whole set by preference
    (which would wrongly promote deepseek)."""
    c = _load()
    disc = _disc_serving("qwen3.5-9b", "deepseek-r1-70b")
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="subagent"))
    assert pick == "qwen3.5-9b", "named members are the priority prefix; the tail follows them"


def test_member_not_displaced_by_idle_tail_under_mild_load():
    """Cross-tier spill happens only on saturation/unavailability, NOT on mild load.
    A lightly-loaded NAMED member must beat an idle NON-MEMBER fallthrough model:
    load-aware spill stays *within* the named members; the open-fallthrough tail is
    a fallback tier, not a load-spill peer. (Brady approved saturation-spill, not
    any-load cross-tier spill — dumping high-quality work onto a 9B because the 35B
    has one in-flight request is a quality regression.)

    subagent member qwen3.5-9b is live at 1/3 slots; trinity-mini (tail) is live and
    idle; qwen3.5-35b is down -> must pick the loaded member 9b, not the idle tail.
    """
    from llm_relay.routing.keys import compose_backend_key
    c = _load()
    disc = DiscoveryManager()
    # qwen3.5-9b: a NAMED subagent member, live, lightly loaded (1 of 3 slots).
    k9 = compose_backend_key("local-llm", 8080, "")
    s9 = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["qwen3.5-9b"])
    c9 = EndpointClient(provider_name="local-llm", base_url="x", state=s9, max_concurrent=3)
    c9.inflight_used = 1
    disc.clients[k9] = c9
    disc.model_to_client["qwen3.5-9b"] = k9
    # trinity-mini: NOT a subagent member (open-fallthrough tail), live, idle.
    disc.clients["k::trinity"] = EndpointClient(
        provider_name="local-llm", base_url="x",
        state=EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=["trinity-mini"]))
    disc.model_to_client["trinity-mini"] = "k::trinity"

    sel = ModelSelector(c, disc)
    assert sel.select_best(RoutingContext(requested_model="subagent")) == "qwen3.5-9b", \
        "a lightly-loaded named member must beat an idle non-member tail (no mild-load cross-tier spill)"


def test_explicit_model_does_not_fall_through_to_fleet():
    """Fallthrough is for ALIASES (the use-case front door), NOT explicit/host-pinned
    requests — that tier is deliberately specific. A strict explicit request for a
    DOWN model must still yield no candidate even when other models are live."""
    c = _load()  # explicit.strict is True in the test policy
    disc = _disc_serving("trinity-mini")  # the requested model is not live; trinity is
    sel = ModelSelector(c, disc)
    pick = sel.select_best(RoutingContext(requested_model="qwen3.5-9b"))
    assert pick is None, "explicit/strict requests must not fall through to the fleet"


def test_unknown_model_routes_over_gguf_reported_fleet():
    """The unknown-model fallback (an unrecognized id -> best available live model)
    must include GGUF-reported (llama.cpp) backends — the same fix as the fallthrough
    tail. The backend reports 'Trinity-Mini-...gguf'; an unknown request must resolve
    to trinity-mini (matched to the config name via get_model_state), not None."""
    c = _load()
    disc = DiscoveryManager()
    state = EndpointState(
        provider="local-llm", status=EndpointStatus.healthy,
        models=["Trinity-Mini-UD-Q4_K_XL.gguf"],  # GGUF filename, not the config name
    )
    disc.clients["k::t"] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client["trinity-mini"] = "k::t"
    sel = ModelSelector(c, disc)
    assert sel.select_best(RoutingContext(requested_model="some-unrecognized-model-xyz")) == "trinity-mini", \
        "unknown-model fallback must reach GGUF-reported backends (matched by config name)"


# ---------------------------------------------------------------------------
# Context-fit contract: when a request can't fit ANY live model, the relay
# distinguishes oversize-for-now (a big-enough model exists in the catalog but is
# down -> wait) from oversize-period (nothing in the fleet is big enough -> resize
# or defer). The relay knows every model's window, so it can compute which.
# ---------------------------------------------------------------------------

def test_diagnose_context_shortfall_oversize_for_now():
    """Request needs 100k; only qwen3.5-9b (32768) is live, but qwen3.5-35b (262144)
    exists in the catalog (just down) -> 'oversize_for_now' (waiting for a big-enough
    model to return is viable)."""
    c = _load()
    disc = _disc_serving("qwen3.5-9b")
    sel = ModelSelector(c, disc)
    diag = sel.diagnose_context_shortfall(RoutingContext(requested_model="main", min_context=100000))
    assert diag is not None
    assert diag["classification"] == "oversize_for_now"
    assert diag["max_available_now"] == 32768
    assert diag["max_in_catalog"] == 262144
    assert diag["estimated_tokens"] == 100000


def test_diagnose_context_shortfall_oversize_period():
    """Request needs 300k, beyond EVERY catalog model's window (max local 262144)
    -> 'oversize_period': waiting cannot help; the client must resize or defer."""
    c = _load()
    disc = _disc_serving("qwen3.5-9b")
    sel = ModelSelector(c, disc)
    diag = sel.diagnose_context_shortfall(RoutingContext(requested_model="main", min_context=300000))
    assert diag is not None
    assert diag["classification"] == "oversize_period"


def test_diagnose_context_shortfall_none_when_a_live_model_fits():
    """A live model can hold the request -> no shortfall. qwen3.5-9b (32768) is live
    and the request needs only 20k."""
    c = _load()
    disc = _disc_serving("qwen3.5-9b")
    sel = ModelSelector(c, disc)
    assert sel.diagnose_context_shortfall(RoutingContext(requested_model="main", min_context=20000)) is None


def test_diagnose_context_shortfall_none_when_nothing_live():
    """Nothing live at all -> an availability problem, not a context one; the
    diagnosis is None and the generic 503 stands."""
    c = _load()
    disc = DiscoveryManager()
    sel = ModelSelector(c, disc)
    assert sel.diagnose_context_shortfall(RoutingContext(requested_model="main", min_context=100000)) is None
