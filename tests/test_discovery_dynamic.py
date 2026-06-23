"""Dynamic model discovery: a model on a provider's `discover_ports` that has no
models.yaml entry is auto-registered as a runtime-discovered model — name-routable
only (manual_only), never persisted, dropped when its port stops reporting it.

`_reconcile_discovered` is the synchronous core (the lifespan task just calls it on
a poll cadence); these tests drive it directly with a stub discover-port client and
then exercise the selector exactly like test_selector.py does. Placeholder names
only (`local-llm`, `127.0.0.1`)."""
from __future__ import annotations

from pathlib import Path

from llm_relay.api.app import (
    _build_available_payload,
    _build_model_card,
    _reconcile_discovered,
    create_app,
)
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus, ModelStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.keys import compose_backend_key
from llm_relay.routing.selector import ModelSelector, RoutingContext


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# A discover port that is NOT any static model's port in the test config.
DISCOVER_PORT = 9999
DISCOVER_KEY = compose_backend_key("local-llm", DISCOVER_PORT, "")
KEY_META = {DISCOVER_KEY: ("local-llm", DISCOVER_PORT)}


def _load() -> ConfigLoader:
    c = ConfigLoader(config_dir=CONFIG_DIR)
    c.load()
    return c


def _seed_discover_client(disc: DiscoveryManager, reported: list[str]) -> None:
    """Place a healthy discover-port client under DISCOVER_KEY reporting `reported`
    in /v1/models (mirrors what the poll loop sets on a real discover backend)."""
    disc.clients[DISCOVER_KEY] = EndpointClient(
        provider_name="local-llm",
        base_url="http://127.0.0.1",
        state=EndpointState(
            provider="local-llm", status=EndpointStatus.healthy, models=reported
        ),
    )


def test_reconcile_registers_discovered_model():
    """A discover-port backend reporting an unconfigured id `bakeoff` becomes a
    runtime model: present in the registry, flagged discovered + manual_only,
    mapped to the discover client, and resolvable as available."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])

    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    assert "bakeoff" in c.models.models
    m = c.models.models["bakeoff"]
    assert m.discovered is True
    assert m.manual_only is True
    assert m.provider == "local-llm" and m.port == DISCOVER_PORT
    assert disc.model_to_client["bakeoff"] == DISCOVER_KEY
    assert disc.get_model_state("bakeoff") == ModelStatus.available


def test_reconcile_is_idempotent_for_a_still_present_model():
    """A second reconcile while the model is still reported must not duplicate or
    disturb it — the registry entry and mapping are unchanged."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])

    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)
    first = c.models.models["bakeoff"]
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    assert c.models.models["bakeoff"] is first, "still-present model must not be rebuilt"
    assert disc.model_to_client["bakeoff"] == DISCOVER_KEY


def test_discovered_model_is_routable_by_exact_name():
    """An exact-name request for the discovered model selects it: the exact-name
    branch routes to it despite manual_only (the isolation flag only holds it out
    of AUTO-selection surfaces)."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    sel = ModelSelector(c, disc)
    assert sel.select_best(RoutingContext(requested_model="bakeoff")) == "bakeoff"


def test_discovered_model_is_isolated_from_alias_fallthrough():
    """A discovered model must NOT be reachable via alias open-fallthrough: it
    carries manual_only, so it is held out of the tail. With every member of `main`
    down and only `bakeoff` live, an alias request must NOT route to it (it would,
    were it not isolated — open fallthrough otherwise reaches any live model)."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])  # only the discovered model is live
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    sel = ModelSelector(c, disc)
    # Not in the open-fallthrough candidate set for an alias...
    cands, _ = sel._build_candidates(RoutingContext(requested_model="main"))
    assert "bakeoff" not in cands, "discovered/manual_only must stay out of the alias tail"
    # ...nor in the unknown-id open ranking over the live fleet.
    cands_u, ordered_u = sel._build_candidates(RoutingContext(requested_model="totally-unknown-zzz"))
    assert ordered_u is False and "bakeoff" not in cands_u
    # ...so an alias request whose members are all down does NOT fall through to it.
    assert sel.select_best(RoutingContext(requested_model="main")) is None, \
        "alias must not select a discovered model merely because it is the only thing live"


def test_reconcile_leaves_static_model_untouched():
    """A statically-configured model on a normal port must be neither removed nor
    altered by reconcile — only discovered entries are managed. The discover port
    also reports `qwen3.5-9b`'s served name, which must defer to the static entry
    (never shadow a configured model)."""
    c = _load()
    static_before = c.models.models["qwen3.5-9b"]
    disc = DiscoveryManager()
    # Discover backend happens to also report a configured model's id alongside the
    # ad-hoc one; the configured model must win and bakeoff still gets registered.
    _seed_discover_client(disc, ["bakeoff", "qwen3.5-9b"])

    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    assert c.models.models["qwen3.5-9b"] is static_before, "static model must be untouched"
    assert c.models.models["qwen3.5-9b"].discovered is False
    # The static model's mapping must NOT be hijacked by the discover key.
    assert disc.model_to_client.get("qwen3.5-9b") != DISCOVER_KEY
    assert "bakeoff" in c.models.models and c.models.models["bakeoff"].discovered is True


def test_reconcile_removes_stale_discovered_model_when_port_goes_quiet():
    """Lifecycle: once the discover port stops reporting the model (empty /v1/models),
    a second reconcile drops it from the registry, the model->client map, and the
    served-name map — so a transient bake-off model doesn't linger as a dead route."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)
    assert "bakeoff" in c.models.models

    # Port now reports nothing (model unloaded / box repurposed).
    disc.clients[DISCOVER_KEY].state.models = []
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    assert "bakeoff" not in c.models.models
    assert "bakeoff" not in disc.model_to_client
    assert "bakeoff" not in disc.served_names


def test_reconcile_skips_missing_client():
    """A discover key with no registered client yet (poll not started) is skipped
    cleanly — no crash, nothing registered."""
    c = _load()
    disc = DiscoveryManager()  # DISCOVER_KEY intentionally absent from .clients
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)
    assert not any(m.discovered for m in c.models.models.values())


def test_discovered_model_flagged_in_available_payload_and_card():
    """The discovered flag is surfaced (additively) on /v1/available-models and on
    the per-model card, so the cockpit can distinguish an ad-hoc model; a normal
    configured model carries no such flag."""
    c = _load()
    disc = DiscoveryManager()
    _seed_discover_client(disc, ["bakeoff"])
    _reconcile_discovered(c, disc, {DISCOVER_KEY}, KEY_META)

    payload = _build_available_payload(c, disc)
    assert payload["bakeoff"]["discovered"] is True
    # Additive: a configured model must not gain the key.
    assert "discovered" not in payload["qwen3.5-9b"]

    card = _build_model_card(c, disc, "bakeoff")
    assert card is not None and card["discovered"] is True
    configured_card = _build_model_card(c, disc, "qwen3.5-9b")
    assert "discovered" not in configured_card


def test_loader_parses_discover_ports(tmp_path):
    """providers.yaml `discover_ports` is parsed onto ProviderConfig (and omitted
    defaults to an empty list)."""
    (tmp_path / "providers.yaml").write_text(
        "providers:\n"
        "  local-llm:\n    type: openai\n    base_url: http://127.0.0.1\n    discover_ports: [9001, 9002]\n"
        "  other:\n    type: openai\n    base_url: http://127.0.0.1\n"
    )
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    assert c.providers["local-llm"].discover_ports == [9001, 9002]
    assert c.providers["other"].discover_ports == []


async def test_lifespan_registers_discover_backend_and_reconcile_task(tmp_path):
    """End-to-end wiring: with a provider declaring discover_ports, the lifespan
    registers a bare polling client for that port (keyed compose_backend_key,
    empty models_hint) and spawns the reconcile task. A static model's own port is
    registered normally (not double-registered). Drive the lifespan context
    directly (httpx's ASGITransport does not run lifespan events); on exit the
    context's shutdown cancels every spawned task. No backend listens on the port,
    so the poll loop just reads unavailable — harmless; we assert on registration."""
    (tmp_path / "providers.yaml").write_text(
        "providers:\n"
        "  local-llm:\n"
        "    type: openai\n"
        "    base_url: http://127.0.0.1\n"
        "    enabled: true\n"
        "    poll_interval: 30s\n"
        "    discover_ports: [9101]\n"
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  static-a:\n    provider: local-llm\n    port: 8080\n    use_cases: {fast: 1}\n"
    )
    app = create_app(config_dir=tmp_path)
    disc = app.state.discovery
    discover_key = compose_backend_key("local-llm", 9101, "")
    static_key = compose_backend_key("local-llm", 8080, "")

    async with app.router.lifespan_context(app):
        assert discover_key in disc.clients, "discover_port must register a polling client"
        assert static_key in disc.clients, "static model's port still registered normally"
        # The discover client carries no model hint (its models are reconciled in).
        assert not any(k == discover_key for k in disc.model_to_client.values()), \
            "discover backend starts with an empty models_hint"
        # The reconcile task was spawned alongside the poll loops (2 poll loops +
        # 1 reconcile loop).
        assert len(disc._tasks) >= 3
    # Shutdown cancelled everything, including the reconcile loop.
    assert disc._tasks == []
