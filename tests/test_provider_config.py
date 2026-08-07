"""Verify max_concurrent loads from providers.yaml into ProviderConfig."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_relay.config.loader import ConfigLoader


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        (3, 3),       # explicit value carries through
        (None, None), # omitted key defaults to None
    ],
    ids=["explicit-3", "omitted"],
)
def test_loader_max_concurrent_round_trip(tmp_path: Path, yaml_value, expected):
    """Verify max_concurrent loads from providers.yaml and respects defaults."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    provider_cfg = {
        "type": "openai",
        "base_url": "http://127.0.0.1",
        "ownership": "ciq_owned",
        "enabled": True,
    }
    if yaml_value is not None:
        provider_cfg["max_concurrent"] = yaml_value
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": provider_cfg}
    }))

    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()

    assert loader.providers["local-llm"].max_concurrent == expected


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        (45.0, 45.0),  # explicit value carries through
        (None, 30.0),  # omitted key defaults to 30.0
    ],
    ids=["explicit-45", "omitted-default-30"],
)
def test_loader_slot_wait_timeout_round_trip(tmp_path: Path, yaml_value, expected):
    """Verify slot_wait_timeout loads from providers.yaml and defaults to 30.0."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    provider_cfg = {
        "type": "openai",
        "base_url": "http://127.0.0.1",
        "ownership": "ciq_owned",
        "enabled": True,
    }
    if yaml_value is not None:
        provider_cfg["slot_wait_timeout"] = yaml_value
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": provider_cfg}
    }))

    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()

    assert loader.providers["local-llm"].slot_wait_timeout == expected


def test_vendor_cloud_provider_type_is_rejected(tmp_path: Path):
    """`type: anthropic` must fail to load.

    The gateway serves CIQ-operated inference and does not proxy vendor clouds
    (decision 2026-07-31). `ProviderType.anthropic` was removed so that
    reintroducing a vendor cloud is a code change and a review, not a quiet
    config edit — this pins that. It also documents that the enum was always
    decorative: nothing in the relay branches on `type`, so an anthropic-typed
    provider was forwarded as OpenAI chat-completions anyway.
    """
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "anthropic": {
                "type": "anthropic",
                "base_url": "https://api.anthropic.com",
                "ownership": "third_party",
            }
        }
    }))
    loader = ConfigLoader(config_dir=cfg_dir)
    with pytest.raises(ValueError):
        loader.load()


def test_openai_provider_type_still_loads(tmp_path: Path):
    """Control for the above: the one supported wire protocol is unaffected."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {
                "type": "openai",
                "base_url": "http://127.0.0.1",
                "ownership": "ciq_owned",
            }
        }
    }))
    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()
    assert loader.providers["local-llm"].type.value == "openai"
