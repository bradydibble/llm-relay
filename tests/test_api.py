"""API payload builder tests (no live polling)."""
from __future__ import annotations

from pathlib import Path

from llm_relay.api.app import _build_available_payload
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg() -> ConfigLoader:
    c = ConfigLoader(config_dir=CONFIG_DIR)
    c.load()
    return c


def _seed(disc: DiscoveryManager, model: str, status: EndpointStatus) -> None:
    """Wire `model` into discovery so get_model_state returns the given status."""
    key = f"k:{model}"
    state = EndpointState(provider="local-llm", status=status, models=[model])
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


def test_alias_info_skips_unavailable_first_member():
    """If the first member is unavailable but a later one is healthy, current
    should be that later one — matching the selector's behavior."""
    cfg = _load_cfg()
    disc = DiscoveryManager()
    _seed(disc, "qwen3.5-35b", EndpointStatus.unavailable)
    _seed(disc, "qwen3.5-9b", EndpointStatus.healthy)

    info = _build_available_payload(cfg, disc)["alias_info"]["main"]
    assert info["current"] == "qwen3.5-9b"
    assert info["context_window"] == cfg.models.models["qwen3.5-9b"].context_window


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
