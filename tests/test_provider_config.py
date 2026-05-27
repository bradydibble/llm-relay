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
    provider_cfg = {"type": "openai", "base_url": "http://127.0.0.1", "enabled": True}
    if yaml_value is not None:
        provider_cfg["max_concurrent"] = yaml_value
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"local-llm": provider_cfg}
    }))

    loader = ConfigLoader(config_dir=cfg_dir)
    loader.load()

    assert loader.providers["local-llm"].max_concurrent == expected
