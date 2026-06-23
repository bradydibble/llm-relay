"""Deterministic dispatcher (plan 3): logical-model routing + intent + decision."""
from __future__ import annotations

from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.selector import ModelSelector, RoutingContext


PROVIDERS = (
    "providers:\n"
    "  p1:\n    base_url: http://127.0.0.1\n"
    "  p2:\n    base_url: http://127.0.0.1\n"
)
MODELS = (
    "models:\n"
    "  v-hi:\n    provider: p1\n    port: 8001\n    logical: big\n    quant: q8\n    preference: 0.9\n"
    "  v-lo:\n    provider: p2\n    port: 8002\n    logical: big\n    quant: q4\n    preference: 0.5\n"
)


def _cfg(tmp_path, models_yaml=MODELS, providers_yaml=PROVIDERS) -> ConfigLoader:
    (tmp_path / "providers.yaml").write_text(providers_yaml)
    (tmp_path / "models.yaml").write_text(models_yaml)
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    return c


def _seed(disc: DiscoveryManager, model: str, provider: str, status=EndpointStatus.healthy) -> None:
    key = f"k:{model}"
    disc.clients[key] = EndpointClient(
        provider_name=provider, base_url="x",
        state=EndpointState(provider=provider, status=status, models=[model]),
    )
    disc.model_to_client[model] = key


# --- Task 1: logical-model routing -----------------------------------------

def test_logical_routes_to_preferred_variant(tmp_path):
    cfg = _cfg(tmp_path)
    disc = DiscoveryManager()
    _seed(disc, "v-hi", "p1")
    _seed(disc, "v-lo", "p2")
    sel = ModelSelector(cfg, disc)
    assert sel.select_best(RoutingContext(requested_model="big")) == "v-hi"


def test_logical_falls_to_other_variant_when_preferred_down(tmp_path):
    cfg = _cfg(tmp_path)
    disc = DiscoveryManager()
    _seed(disc, "v-lo", "p2")  # only the low-preference variant is up
    sel = ModelSelector(cfg, disc)
    assert sel.select_best(RoutingContext(requested_model="big")) == "v-lo"


def test_logical_none_when_no_variant_live(tmp_path):
    cfg = _cfg(tmp_path)
    sel = ModelSelector(cfg, DiscoveryManager())
    assert sel.select_best(RoutingContext(requested_model="big")) is None


def test_logical_excludes_manual_only_variants(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  v-auto:\n    provider: p1\n    port: 8001\n    logical: big\n    preference: 0.6\n"
        "  v-manual:\n    provider: p2\n    port: 8002\n    logical: big\n    preference: 0.9\n    manual_only: true\n",
    )
    disc = DiscoveryManager()
    _seed(disc, "v-auto", "p1")
    _seed(disc, "v-manual", "p2")
    sel = ModelSelector(cfg, disc)
    # v-manual has higher preference but is manual_only -> excluded from logical routing.
    assert sel.select_best(RoutingContext(requested_model="big")) == "v-auto"


def test_concrete_variant_name_not_shadowed_by_logical(tmp_path):
    cfg = _cfg(tmp_path)
    disc = DiscoveryManager()
    _seed(disc, "v-hi", "p1")
    sel = ModelSelector(cfg, disc)
    # Requesting the concrete variant by name routes to it, not via the logical pipeline.
    assert sel.select_best(RoutingContext(requested_model="v-hi")) == "v-hi"


def test_logical_branch_does_not_perturb_alias_or_concrete(tmp_path):
    """No-regression: with the logical branch present, an existing alias and an
    existing concrete request resolve to the same candidates as before."""
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  a9:\n    provider: p1\n    port: 8001\n    use_cases: {fast: 1}\n    preference: 0.7\n"
        "  v1:\n    provider: p1\n    port: 8003\n    logical: big\n"
        "  v2:\n    provider: p2\n    port: 8004\n    logical: big\n",
    )
    disc = DiscoveryManager()
    for m, p in [("a9", "p1"), ("v1", "p1"), ("v2", "p2")]:
        _seed(disc, m, p)
    sel = ModelSelector(cfg, disc)
    # Alias 'fast' still resolves to its member first (ordered priority), unaffected.
    cands, ordered = sel._build_candidates(RoutingContext(requested_model="fast"))
    assert cands[0] == "a9" and ordered is True
    # Concrete 'a9' still resolves to itself first.
    cands2, _ = sel._build_candidates(RoutingContext(requested_model="a9"))
    assert cands2[0] == "a9"
    # Logical 'big' resolves to its variants (the new path).
    cands3, _ = sel._build_candidates(RoutingContext(requested_model="big"))
    assert set(cands3) == {"v1", "v2"}


# --- Task 2: quality floor (request) ---------------------------------------

def test_quality_floor_filters_low_preference_variants(tmp_path):
    cfg = _cfg(tmp_path)  # v-hi pref 0.9, v-lo pref 0.5
    disc = DiscoveryManager()
    _seed(disc, "v-hi", "p1")
    _seed(disc, "v-lo", "p2")
    sel = ModelSelector(cfg, disc)
    # floor 0.6 drops v-lo (0.5); only v-hi qualifies.
    assert sel.select_best(RoutingContext(requested_model="big", min_preference=0.6)) == "v-hi"
    # floor 0.95 drops both.
    assert sel.select_best(RoutingContext(requested_model="big", min_preference=0.95)) is None


# --- Task 3: decision record (quant / node / batch) ------------------------

def test_decision_record_carries_quant_node_batch(tmp_path):
    from llm_relay.routing.router import _candidate_to_route_result

    cfg = _cfg(tmp_path)
    disc = DiscoveryManager()
    _seed(disc, "v-hi", "p1")
    sel = ModelSelector(cfg, disc)
    ctx = RoutingContext(requested_model="big", sla_class="agentic", urgency="low")
    chain = sel.select_chain(ctx)
    assert chain and chain[0].model == "v-hi" and chain[0].quant == "q8"
    rr = _candidate_to_route_result(chain[0], ctx)
    assert rr.decision["quant"] == "q8"
    assert rr.decision["node"] == "p1"
    assert rr.decision["batch"] == "coalesce"  # agentic class
    assert rr.decision["sla_class"] == "agentic"
    assert rr.decision["urgency"] == "low"


def test_batch_policy_mapping():
    from llm_relay.routing.selector import batch_policy_for

    assert batch_policy_for("interactive") == "none"
    assert batch_policy_for("agentic") == "coalesce"
    assert batch_policy_for("bulk") == "coalesce"
    assert batch_policy_for(None) == "none"


def test_flat_model_decision_has_null_quant(tmp_path):
    """A flat model with no quant must not crash the decision record."""
    from llm_relay.routing.router import _candidate_to_route_result

    cfg = _cfg(tmp_path, "models:\n  flat:\n    provider: p1\n    port: 8001\n")
    disc = DiscoveryManager()
    _seed(disc, "flat", "p1")
    sel = ModelSelector(cfg, disc)
    chain = sel.select_chain(RoutingContext(requested_model="flat"))
    rr = _candidate_to_route_result(chain[0], RoutingContext(requested_model="flat"))
    assert rr.decision["quant"] is None
    assert rr.decision["node"] == "p1"
