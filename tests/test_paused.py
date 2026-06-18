"""Maintenance-pause: discovery state, expiry, routing skip, persistence, payload.

The /admin/pause and /admin/resume HTTP routes are thin wrappers over
DiscoveryManager.pause_provider/resume_provider + ConfigLoader.save_paused_providers
(all covered here); they are smoke-tested live against the running relay.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_relay.api.app import _build_available_payload
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.selector import ModelSelector, RoutingContext

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg() -> ConfigLoader:
    c = ConfigLoader(config_dir=CONFIG_DIR)
    c.load()
    return c


def _seed(disc: DiscoveryManager, model: str, provider: str,
          status: EndpointStatus = EndpointStatus.healthy) -> None:
    key = f"k:{model}"
    disc.clients[key] = EndpointClient(
        provider_name=provider, base_url="x",
        state=EndpointState(provider=provider, status=status, models=[model]))
    disc.model_to_client[model] = key


def test_pause_resume_mutates_state():
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    assert disc.is_provider_paused("p1") is False
    disc.pause_provider("p1", until=None, reason="maint")
    assert disc.is_provider_paused("p1") is True
    assert disc.clients["k:m1"].state.paused_reason == "maint"
    disc.resume_provider("p1")
    assert disc.is_provider_paused("p1") is False


def test_indefinite_pause_stays_paused():
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    disc.pause_provider("p1", until=None)
    assert disc.is_provider_paused("p1") is True


def test_expired_pause_auto_resumes():
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    disc.pause_provider("p1", until=past)
    assert disc.is_provider_paused("p1") is False        # expired
    assert disc.clients["k:m1"].state.paused is False     # and healed


def test_naive_future_timestamp_stays_paused():
    # Naive timestamps are interpreted as UTC (the scheduler always sends aware
    # UTC; this guards the naive-normalization path). Build the naive string from
    # UTC so it's unambiguously future regardless of the box's local zone.
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    future_naive = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None).isoformat()
    disc.pause_provider("p1", until=future_naive)
    assert disc.is_provider_paused("p1") is True


def test_unparseable_until_stays_paused():
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    disc.pause_provider("p1", until="not-a-timestamp")
    assert disc.is_provider_paused("p1") is True


def test_pause_does_not_touch_circuit_breaker():
    disc = DiscoveryManager(); _seed(disc, "m1", "p1")
    st = disc.clients["k:m1"].state
    disc.pause_provider("p1")
    assert st.consecutive_failures == 0 and st.circuit_open is False


def test_selector_skips_paused_provider():
    cfg = _load_cfg()
    model = "qwen3.5-35b"
    provider = cfg.models.models[model].provider
    disc = DiscoveryManager(); _seed(disc, model, provider)
    sel = ModelSelector(cfg, disc)
    assert sel.select_best(RoutingContext(requested_model=model)) == model
    assert any(c.model == model for c in sel.select_chain(RoutingContext(requested_model=model)))
    disc.pause_provider(provider)
    assert sel.select_best(RoutingContext(requested_model=model)) is None
    assert all(c.model != model for c in sel.select_chain(RoutingContext(requested_model=model)))


def test_available_models_marks_paused():
    cfg = _load_cfg()
    model = "qwen3.5-35b"
    provider = cfg.models.models[model].provider
    disc = DiscoveryManager(); _seed(disc, model, provider)
    assert _build_available_payload(cfg, disc)[model]["status"] != "paused"
    disc.pause_provider(provider, until=None, reason="maint")
    payload = _build_available_payload(cfg, disc)
    assert payload[model]["status"] == "paused"


def test_loader_paused_roundtrip(tmp_path):
    c = ConfigLoader(config_dir=tmp_path)
    assert c.load_paused_providers() == {}                      # missing file -> {}
    c.save_paused_providers({"p1": {"until": None, "reason": "x"}})
    assert c.load_paused_providers() == {"p1": {"until": None, "reason": "x"}}
    c.save_paused_providers({})                                  # resume clears
    assert c.load_paused_providers() == {}


def test_loader_invalid_json_returns_empty(tmp_path):
    (tmp_path / "paused-providers.json").write_text("{ not json")
    assert ConfigLoader(config_dir=tmp_path).load_paused_providers() == {}
