"""API payload builder tests (no live polling)."""
from __future__ import annotations

from pathlib import Path

import httpx

from llm_relay.api.app import (
    _build_available_payload,
    _build_model_card,
    _build_models_list_payload,
    _resolve_context_window,
    create_app,
)
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg() -> ConfigLoader:
    c = ConfigLoader(config_dir=CONFIG_DIR)
    c.load()
    return c


def _seed(
    disc: DiscoveryManager,
    model: str,
    status: EndpointStatus,
    live_ctx: int | None = None,
) -> None:
    """Wire `model` into discovery so get_model_state returns the given status.

    Pass live_ctx to simulate a backend reporting max_model_len for `model`
    (mirrors a vLLM /v1/models response), so live-context resolution can be
    exercised without a real probe."""
    key = f"k:{model}"
    state = EndpointState(provider="local-llm", status=status, models=[model])
    if live_ctx is not None:
        state.model_max_lens = {model: live_ctx}
    disc.clients[key] = EndpointClient(provider_name="local-llm", base_url="x", state=state)
    disc.model_to_client[model] = key


def test_payload_includes_alias_info_for_every_alias():
    cfg = _load_cfg()
    disc = DiscoveryManager()
    payload = _build_available_payload(cfg, disc)
    assert "alias_info" in payload
    for alias in cfg.models.aliases.keys():
        assert alias in payload["alias_info"], f"missing alias_info for {alias}"
        entry = payload["alias_info"][alias]
        assert "members" in entry and "current" in entry and "context_window" in entry


def test_alias_info_picks_first_available_member():
    """`main` is [qwen3.5-35b, deepseek-r1-70b, llama-3.3-70b, qwen3.5-9b]. With only
    the 35b healthy, current should be qwen3.5-35b and ctx its 262144."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    _seed(disc, "qwen3.5-35b", EndpointStatus.healthy)

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["current"] == "qwen3.5-35b"
    assert info["context_window"] == cfg.models.models["qwen3.5-35b"].context_window


def test_alias_info_current_tracks_first_available_member():
    """`current` still reflects the live-serving member: when the first member
    is unavailable but a later one is healthy, current is that later one."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    _seed(disc, "qwen3.5-35b", EndpointStatus.unavailable)
    _seed(disc, "qwen3.5-9b", EndpointStatus.healthy)

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["current"] == "qwen3.5-9b"


def test_alias_info_context_window_is_the_live_servable_ceiling():
    """context_window advertises what the alias can ACTUALLY serve right now — the
    largest live window among its available candidates — not the down primary's
    nominal window.

    Supersedes the old "advertise the primary member, stable" behavior: advertising
    262144 while a 9b (32768) is what actually serves is the "sized it right, still
    503'd" lie. Open-by-default makes fallback-serving the norm (not a brief blip),
    so the advertised ceiling must track the live fleet — the number a client can
    safely size a request up to."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    _seed(disc, "qwen3.5-35b", EndpointStatus.unavailable)
    _seed(disc, "qwen3.5-9b", EndpointStatus.healthy)

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["current"] == "qwen3.5-9b", "current reflects the live-serving fallback"
    assert info["context_window"] == cfg.models.models["qwen3.5-9b"].context_window, \
        "context_window must report the live-servable ceiling (9b 32768), not the down primary (35b)"


def test_alias_info_context_window_reflects_open_fallthrough_tail():
    """When ALL named members are down but a non-member is live (open fallthrough),
    the advertised ceiling tracks the tail model that would actually serve — proving
    the ceiling is computed over the live fleet, not just the named members. main's
    members are all down; trinity-mini (16384, not a member) is the only live model."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    _seed(disc, "trinity-mini", EndpointStatus.healthy)  # not a member of `main`

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["context_window"] == cfg.models.models["trinity-mini"].context_window, \
        "ceiling must reflect the live fallthrough model (trinity 16384) the request would route to"


def test_alias_info_falls_back_to_first_declared_when_none_available():
    """When no member is reachable, still report something useful (the first
    declared) so clients don't break or render blank context."""
    cfg = _load_cfg()
    disc = DiscoveryManager()  # nothing seeded — all unavailable

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["current"] == "qwen3.5-35b", "should fall back to first declared member"
    assert info["context_window"] == cfg.models.models["qwen3.5-35b"].context_window


def test_aliases_block_remains_backward_compatible():
    """Existing clients read `aliases[<name>]` as a list[str]; that contract
    must not change."""
    cfg = _load_cfg()
    payload = _build_available_payload(cfg, DiscoveryManager())
    assert isinstance(payload["aliases"], dict)
    for members in payload["aliases"].values():
        assert isinstance(members, list)
        assert all(isinstance(m, str) for m in members)


# ---------------------------------------------------------------------------
# Context-window resolution + OpenAI /v1/models context exposure
# ---------------------------------------------------------------------------

def test_resolve_context_window_concrete_model_prefers_live_over_config():
    cfg = _load_cfg()
    disc = DiscoveryManager()
    # No backend reporting a live value → static config wins.
    assert _resolve_context_window(cfg, disc, "qwen3.5-9b") == \
        cfg.models.models["qwen3.5-9b"].context_window
    # Backend reports a live max_model_len → authoritative, overrides config.
    _seed(disc, "qwen3.5-9b", EndpointStatus.healthy, live_ctx=12345)
    assert _resolve_context_window(cfg, disc, "qwen3.5-9b") == 12345


def test_resolve_context_window_alias_reports_live_servable_ceiling():
    """An alias resolves to the largest context it can SERVE right now (the max
    live window among its available candidates), not its first-declared member's
    nominal window. With nothing live it falls back to the primary's window so the
    advertised capability survives a full-fleet outage."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    # Nothing live → fall back to the primary (first-declared) member's window.
    assert _resolve_context_window(cfg, disc, "main") == cfg.models.models["qwen3.5-35b"].context_window
    # Primary down, smaller fallback up → reports the fallback's (servable) window.
    _seed(disc, "qwen3.5-35b", EndpointStatus.unavailable)
    _seed(disc, "qwen3.5-9b", EndpointStatus.healthy)
    assert _resolve_context_window(cfg, disc, "main") == cfg.models.models["qwen3.5-9b"].context_window


def test_resolve_context_window_unknown_name_is_none():
    cfg = _load_cfg()
    assert _resolve_context_window(cfg, DiscoveryManager(), "no-such-model") is None


def test_v1_models_list_entries_carry_context_length_and_max_model_len():
    """Every /v1/models entry (concrete models and aliases) carries context so
    an OpenAI-compatible client can discover it from the list. Both
    `context_length` and `max_model_len` are present (clients read either)."""
    cfg = _load_cfg()
    payload = _build_models_list_payload(cfg, DiscoveryManager())
    assert payload["object"] == "list"
    by_id = {e["id"]: e for e in payload["data"]}

    nine_b = by_id["local-llm:qwen3.5-9b"]
    assert nine_b["context_length"] == cfg.models.models["qwen3.5-9b"].context_window
    assert nine_b["max_model_len"] == nine_b["context_length"]

    main = by_id["main"]
    assert main["owned_by"] == "llm-relay-alias"
    assert main["context_length"] == cfg.models.models["qwen3.5-35b"].context_window
    assert main["max_model_len"] == main["context_length"]


def test_v1_model_card_for_model_alias_and_unknown():
    cfg = _load_cfg()
    disc = DiscoveryManager()

    model_card = _build_model_card(cfg, disc, "qwen3.5-9b")
    assert model_card["id"] == "qwen3.5-9b"
    assert model_card["owned_by"] == "local-llm"
    assert model_card["context_length"] == cfg.models.models["qwen3.5-9b"].context_window

    alias_card = _build_model_card(cfg, disc, "main")
    assert alias_card["id"] == "main"
    assert alias_card["owned_by"] == "llm-relay-alias"
    assert alias_card["context_length"] == cfg.models.models["qwen3.5-35b"].context_window

    assert _build_model_card(cfg, disc, "definitely-not-a-model") is None


def test_v1_models_list_advertises_qualified_ids():
    """Concrete models are advertised as host-qualified 'provider:model' ids so
    clients can differentiate the same model on different hosts. Bare names are
    not advertised (they still resolve for back-compat); aliases are unchanged."""
    cfg = _load_cfg()
    payload = _build_models_list_payload(cfg, DiscoveryManager())
    by_id = {e["id"]: e for e in payload["data"]}
    ids = set(by_id)
    assert "local-llm:qwen3.5-9b" in ids
    assert "anthropic:claude-3-5-sonnet" in ids
    assert "qwen3.5-9b" not in ids, "bare concrete names must not be advertised"
    assert "main" in ids and "subagent" in ids, "aliases (categories) stay advertised"
    assert by_id["local-llm:qwen3.5-9b"]["owned_by"] == "local-llm"
    # context metadata still resolved (by bare name) and attached to the qualified entry
    assert by_id["local-llm:qwen3.5-9b"]["context_length"] == \
        cfg.models.models["qwen3.5-9b"].context_window


def test_v1_model_card_accepts_qualified_and_bare():
    """The card resolves a qualified id (echoing it) and a bare id (back-compat);
    a mismatched provider:model pair is a 404 (None)."""
    cfg = _load_cfg()
    disc = DiscoveryManager()

    q = _build_model_card(cfg, disc, "local-llm:qwen3.5-9b")
    assert q["id"] == "local-llm:qwen3.5-9b", "card echoes the requested qualified id"
    assert q["owned_by"] == "local-llm"
    assert q["context_length"] == cfg.models.models["qwen3.5-9b"].context_window

    b = _build_model_card(cfg, disc, "qwen3.5-9b")
    assert b["id"] == "qwen3.5-9b", "bare id still resolves and echoes (back-compat)"

    assert _build_model_card(cfg, disc, "anthropic:qwen3.5-9b") is None, \
        "mismatched provider:model is not a valid pairing"


async def test_v1_models_card_route_handles_colon_in_path():
    """Over real HTTP: GET /v1/models/{qualified} captures the colon-bearing id
    (Starlette path param) and serves the card; a mismatched pair 404s. Settles
    that the qualified id survives the routing layer, not just the builder."""
    app = create_app(config_dir=CONFIG_DIR)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        ok = await client.get("/v1/models/local-llm:qwen3.5-9b")
        bad = await client.get("/v1/models/anthropic:qwen3.5-9b")
    assert ok.status_code == 200
    assert ok.json()["id"] == "local-llm:qwen3.5-9b"
    assert bad.status_code == 404


async def test_available_models_legacy_path_is_deprecated_alias():
    """`/available-models` is a deprecated alias of the canonical, OpenAI-namespaced
    `/v1/available-models`. It must still return the identical payload (no client
    breaks) but carry RFC 8594 deprecation headers pointing at the successor; the
    canonical path carries no such header."""
    app = create_app(config_dir=CONFIG_DIR)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        legacy = await client.get("/available-models")
        canonical = await client.get("/v1/available-models")
    assert legacy.status_code == 200 and canonical.status_code == 200
    assert legacy.json() == canonical.json(), "deprecated alias must return the identical payload"
    assert legacy.headers.get("Deprecation") == "true"
    assert "/v1/available-models" in legacy.headers.get("Link", "")
    assert "Deprecation" not in canonical.headers, "canonical path must not be marked deprecated"
