"""Verify max_concurrent loads from providers.yaml into ProviderConfig."""
from __future__ import annotations

from pathlib import Path

import yaml

from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import ProviderConfig, ProviderType


def test_provider_config_has_max_concurrent_field():
    """The field exists with a sensible default (None = unbounded, back-compat)."""
    pc = ProviderConfig(type=ProviderType.openai, base_url="http://x")
    assert hasattr(pc, "max_concurrent")
    assert pc.max_concurrent is None


def test_loader_parses_max_concurrent_when_present(tmp_path: Path):
    """When providers.yaml sets max_concurrent: 3 on a provider, ConfigLoader carries it through."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {
                "type": "openai",
                "base_url": "http://127.0.0.1",
                "enabled": True,
                "max_concurrent": 3,
            }
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({"models": {}, "aliases": {}}))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump({"policy": {}}))

    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()

    assert loader.providers["local-llm"].max_concurrent == 3


def test_loader_max_concurrent_defaults_to_none(tmp_path: Path):
    """When the YAML omits max_concurrent the loaded value is None (unbounded)."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {
            "local-llm": {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True}
        }
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({"models": {}, "aliases": {}}))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump({"policy": {}}))

    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()

    assert loader.providers["local-llm"].max_concurrent is None
