"""Confidentiality axis: workloads only reach borrowed hardware when declared safe.

Ownership is a property of the METAL (a provider), confidentiality a property of
the WORKLOAD (a request). The join is the gate: a `confidential` request — which
is anything that did not explicitly say otherwise — may only be served by
providers marked `ciq_owned`.

Every test here pins one of the four properties the control depends on:
  1. config    — ownership is required and typo-proof
  2. routing   — the filter actually drops third-party metal, fail-closed
  3. terminal  — a mismatch never reads as a retryable outage
  4. plumbing  — the header survives the API allowlist and the key-scope ceiling
"""
import pytest
import yaml

from llm_relay.api.app import _clamp_confidentiality
from llm_relay.auth import Principal
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import (
    Confidentiality,
    EndpointState,
    EndpointStatus,
    Ownership,
)
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.router import _parse_confidentiality
from llm_relay.routing.selector import ModelSelector, RoutingContext

# A two-node fleet: one box CIQ owns, one borrowed. Both serve a tool-capable
# model of identical preference so ONLY ownership can decide between them.
PROVIDERS = {
    "providers": {
        "ciq-box": {
            "type": "openai",
            "base_url": "http://127.0.0.1",
            "ownership": "ciq_owned",
        },
        "borrowed-box": {
            "type": "openai",
            "base_url": "http://127.0.0.2",
            "ownership": "third_party",
        },
    }
}
MODELS = {
    "models": {
        "owned-model": {
            "provider": "ciq-box",
            "port": 8080,
            "preference": 0.8,
            "context_window": 4096,
            "use_cases": {"main": 1},
        },
        "borrowed-model": {
            "provider": "borrowed-box",
            "port": 8081,
            "preference": 0.9,  # HIGHER preference — wins ranking if not filtered
            "context_window": 4096,
            "use_cases": {"main": 2},
        },
    }
}


def _write(tmp_path, providers=PROVIDERS, models=MODELS):
    (tmp_path / "providers.yaml").write_text(yaml.safe_dump(providers))
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(models))
    loader = ConfigLoader(config_dir=tmp_path)
    loader.load()
    return loader


def _disc(*models: str) -> DiscoveryManager:
    """DiscoveryManager reporting each model healthy under an arbitrary key, so
    load-ratio lookups miss and score idle — selection is driven purely by the
    constraint filters under test."""
    disc = DiscoveryManager()
    for m in models:
        state = EndpointState(provider="p", status=EndpointStatus.healthy, models=[m])
        disc.clients[f"k::{m}"] = EndpointClient(provider_name="p", base_url="x", state=state)
        disc.model_to_client[m] = f"k::{m}"
    return disc


# --------------------------------------------------------------------------
# 1. Config: ownership is required and typo-proof
# --------------------------------------------------------------------------

def test_ownership_round_trips_from_yaml(tmp_path):
    cfg = _write(tmp_path)
    assert cfg.providers["ciq-box"].ownership == Ownership.ciq_owned
    assert cfg.providers["borrowed-box"].ownership == Ownership.third_party


def test_missing_ownership_raises_naming_the_provider(tmp_path):
    """A provider with no ownership must refuse to load. Defaulting it either way
    is wrong: guess `ciq_owned` and confidential work silently lands on borrowed
    metal; guess `third_party` and the whole fleet stops serving. Fail at load."""
    bad = {"providers": {"untagged": {"type": "openai", "base_url": "http://127.0.0.1"}}}
    with pytest.raises(ValueError, match="untagged.*ownership"):
        _write(tmp_path, providers=bad)


def test_invalid_ownership_value_raises(tmp_path):
    """A typo must not silently degrade to a permissive default."""
    bad = {
        "providers": {
            "typo": {"type": "openai", "base_url": "http://127.0.0.1", "ownership": "ciq-owned"}
        }
    }
    with pytest.raises(ValueError):
        _write(tmp_path, providers=bad)


# --------------------------------------------------------------------------
# 2. Routing: the filter drops borrowed metal, and fails closed
# --------------------------------------------------------------------------

def test_confidential_request_refuses_third_party_hardware(tmp_path):
    """The borrowed model has HIGHER preference, so without the gate it would win.
    An undeclared request must still land on the CIQ-owned box."""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("owned-model", "borrowed-model"))
    ctx = RoutingContext(requested_model="main")  # no declaration == confidential
    assert ctx.confidentiality == Confidentiality.confidential
    assert sel.select_best(ctx) == "owned-model"


def test_non_confidential_request_may_use_third_party_hardware(tmp_path):
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("owned-model", "borrowed-model"))
    ctx = RoutingContext(
        requested_model="main", confidentiality=Confidentiality.non_confidential
    )
    assert sel.select_best(ctx) == "borrowed-model"


def test_confidential_request_gets_nothing_when_only_borrowed_metal_is_live(tmp_path):
    """The core guarantee: no silent substitution AND no leak. With only borrowed
    hardware up, an undeclared request is refused outright."""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("borrowed-model"))
    assert sel.select_best(RoutingContext(requested_model="main")) is None


def test_ownership_fails_closed_when_provider_is_missing(tmp_path):
    """A model naming a provider that does not exist resolves to third_party, so a
    dangling reference can never widen what confidential work may reach."""
    models = {"models": {"orphan": {"provider": "ghost-box", "port": 9000, "preference": 0.9}}}
    cfg = _write(tmp_path, models=models)
    sel = ModelSelector(cfg, _disc("orphan"))
    assert sel._ownership_of("orphan") == Ownership.third_party
    assert sel.select_best(RoutingContext(requested_model="orphan")) is None


def test_declaring_non_confidential_never_costs_a_ciq_owned_backend(tmp_path):
    """The axis is a ceiling-raise, not a swap: declaring non_confidential widens
    the pool and must never remove CIQ-owned candidates from it."""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("owned-model", "borrowed-model"))
    relaxed = RoutingContext(
        requested_model="main", confidentiality=Confidentiality.non_confidential
    )
    sel._prepare_ranked(relaxed)
    assert "owned-model" in relaxed.filtered


# --------------------------------------------------------------------------
# 3. Terminal, not transient — and actionable
# --------------------------------------------------------------------------

def test_confidentiality_mismatch_is_terminal_not_retryable(tmp_path):
    """Naming a model that lives ONLY on borrowed metal, without declaring, is a
    policy mismatch — nothing in the candidate set can ever satisfy it, so it must
    not be dressed up as a retryable outage. (This is the ornith-397b case.)"""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("borrowed-model"))
    ctx = RoutingContext(requested_model="borrowed-model")
    assert sel.select_chain(ctx) == []
    assert sel.is_transient_no_candidate(ctx) is False


def test_down_ciq_model_is_transient_not_a_confidentiality_block(tmp_path):
    """Counterpart to the above. When a CIQ-owned model WOULD serve and is merely
    down, the remedy is to wait — not to relabel the workload. Misreporting this
    as a policy block would teach callers to declare non_confidential to work
    around ordinary outages, which is precisely the habit that breaks the
    control."""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("borrowed-model"))  # owned-model configured but down
    ctx = RoutingContext(requested_model="main")
    assert sel.select_chain(ctx) == []
    assert sel.is_transient_no_candidate(ctx) is True
    assert sel.diagnose_confidentiality_block(ctx) is None


def test_diagnosis_names_the_hardware_and_the_remedy(tmp_path):
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc("borrowed-model"))
    ctx = RoutingContext(requested_model="borrowed-model")
    sel.select_chain(ctx)
    diag = sel.diagnose_confidentiality_block(ctx)
    assert diag is not None
    assert diag["reason"] == "confidentiality_requires_declaration"
    assert diag["third_party_nodes"] == ["borrowed-box"]
    assert "borrowed-model" in diag["blocked_models"]
    assert "non_confidential" in diag["remedy"]


def test_no_diagnosis_when_caller_already_declared(tmp_path):
    """Confidentiality is not the binding constraint if the caller already opted
    in — a genuine outage must not be misreported as a policy block."""
    cfg = _write(tmp_path)
    sel = ModelSelector(cfg, _disc())  # nothing live at all
    ctx = RoutingContext(
        requested_model="main", confidentiality=Confidentiality.non_confidential
    )
    assert sel.diagnose_confidentiality_block(ctx) is None


# --------------------------------------------------------------------------
# 4. Hardware-pin aliases do not fall through
# --------------------------------------------------------------------------

HW_PIN = {
    "models": {
        "tray-a": {
            "provider": "borrowed-box",
            "port": 8081,
            "preference": 0.99,
            "manual_only": True,
            "use_cases": {"borrowed-tray": 1},
        },
        "owned-model": {
            "provider": "ciq-box",
            "port": 8080,
            "preference": 0.8,
            "use_cases": {"main": 1},
        },
    }
}


def test_hardware_pin_alias_does_not_fall_through_to_other_metal(tmp_path):
    """`borrowed-tray`'s only member is manual_only, so the alias names HARDWARE,
    not a capability. Asking for it must never quietly yield a different box —
    that is the exact failure that motivated this work (a request for the AMD tray
    silently answered by a 35B on the Strix Halo)."""
    cfg = _write(tmp_path, models=HW_PIN)
    sel = ModelSelector(cfg, _disc("owned-model"))  # tray is DOWN, owned box is up
    cands, _ = sel._build_candidates(RoutingContext(requested_model="borrowed-tray"))
    assert cands == ["tray-a"]
    assert "owned-model" not in cands
    assert sel.select_best(RoutingContext(requested_model="borrowed-tray")) is None


def test_capability_alias_keeps_its_open_fallthrough(tmp_path):
    """Regression guard: the hardware-pin exception must not narrow ordinary
    capability aliases, which are deliberately open over the live fleet."""
    cfg = _write(tmp_path, models=HW_PIN)
    sel = ModelSelector(cfg, _disc("owned-model"))
    cands, _ = sel._build_candidates(RoutingContext(requested_model="main"))
    assert "owned-model" in cands


# --------------------------------------------------------------------------
# 5. Header parsing + the key-scope ceiling
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("non_confidential", Confidentiality.non_confidential),
        ("NON_CONFIDENTIAL", Confidentiality.non_confidential),
        ("  non_confidential  ", Confidentiality.non_confidential),
        ("confidential", Confidentiality.confidential),
        ("nonconfidential", Confidentiality.confidential),   # typo -> safe
        ("non-confidential", Confidentiality.confidential),  # typo -> safe
        ("true", Confidentiality.confidential),
        ("", Confidentiality.confidential),
    ],
)
def test_header_parse_fails_closed(raw, expected):
    """Only the exact token opts in. Every other spelling resolves to confidential
    — a governance control must never widen its pool because of a typo."""
    assert _parse_confidentiality({"X-Llm-Relay-Confidentiality": raw}) is expected


def test_header_parse_defaults_confidential_when_absent():
    assert _parse_confidentiality({}) is Confidentiality.confidential


def test_ceiling_clamps_caller_without_third_party_scope():
    h = {"X-Llm-Relay-Confidentiality": "non_confidential"}
    _clamp_confidentiality(Principal(id="jdoe"), True, h)
    assert h["X-Llm-Relay-Confidentiality"] == "confidential"


def test_ceiling_honors_caller_with_third_party_scope():
    h = {"X-Llm-Relay-Confidentiality": "non_confidential"}
    _clamp_confidentiality(Principal(id="lab", scopes=["third_party"]), True, h)
    assert h["X-Llm-Relay-Confidentiality"] == "non_confidential"


def test_ceiling_is_noop_when_auth_disabled():
    """Trusted-listener / open-deployment behavior, matching _clamp_privacy."""
    h = {"X-Llm-Relay-Confidentiality": "non_confidential"}
    _clamp_confidentiality(Principal(id="anonymous"), False, h)
    assert h["X-Llm-Relay-Confidentiality"] == "non_confidential"


def test_ceiling_clamps_none_principal():
    h = {"X-Llm-Relay-Confidentiality": "non_confidential"}
    _clamp_confidentiality(None, True, h)
    assert h["X-Llm-Relay-Confidentiality"] == "confidential"


# --------------------------------------------------------------------------
# 6. End-to-end header plumbing through the API allowlist
#
# /v1/chat/completions forwards only an ALLOWLISTED set of X-Llm-Relay-* headers
# to the router. A header parsed by the router but absent from that list is
# silently dropped and the whole axis ships inert — which is exactly how
# X-Llm-Relay-Candidate-Lane shipped dead. There was no test for this on any
# routing header; these two are it.
# --------------------------------------------------------------------------

def _api_config(tmp_path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump(PROVIDERS))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(MODELS))
    return cfg_dir


async def _captured_headers_for(tmp_path, monkeypatch, send_headers):
    """POST a chat completion and return the headers dict the router actually saw."""
    import httpx
    from httpx import ASGITransport

    from llm_relay.api.app import create_app

    app = create_app(config_dir=_api_config(tmp_path))
    seen = {}

    async def _capture(request_data, headers=None, stream=False):
        seen.update(headers or {})
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(app.state.router, "route_and_forward", _capture)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "main", "messages": [{"role": "user", "content": "hi"}]},
            headers=send_headers,
        )
    return seen


async def test_confidentiality_header_reaches_the_router(tmp_path, monkeypatch):
    """The declaration must survive the allowlist. If this fails, the axis is
    inert no matter how correct the selector is."""
    seen = await _captured_headers_for(
        tmp_path, monkeypatch, {"X-Llm-Relay-Confidentiality": "non_confidential"}
    )
    assert seen.get("X-Llm-Relay-Confidentiality") == "non_confidential"


async def test_absent_header_leaves_router_at_confidential_default(tmp_path, monkeypatch):
    seen = await _captured_headers_for(tmp_path, monkeypatch, {})
    assert "X-Llm-Relay-Confidentiality" not in seen
    assert _parse_confidentiality(seen) is Confidentiality.confidential
