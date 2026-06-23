"""Variant registry: logical-model grouping + per-host exclusivity (plan 2).

Additive data model over the existing flat ``config.models.models`` dict. No
routing-behavior change here; the dispatcher that consumes it is plan 3.
"""
from __future__ import annotations

from llm_relay.config.loader import ConfigLoader


def _cfg(tmp_path, models_yaml: str, providers_yaml: str = "providers: {}\n") -> ConfigLoader:
    (tmp_path / "providers.yaml").write_text(providers_yaml)
    (tmp_path / "models.yaml").write_text(models_yaml)
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    return c


# --- Task 1: variant fields + logical-model derivation ---------------------

def test_variant_fields_parsed(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  m-awq:\n    provider: p1\n    port: 8000\n    logical: qwen14\n    quant: awq-int4\n"
        "  m-q4:\n    provider: p2\n    port: 8000\n    logical: qwen14\n    quant: q4_k_m\n",
    )
    assert cfg.models.models["m-awq"].logical == "qwen14"
    assert cfg.models.models["m-awq"].quant == "awq-int4"


def test_logical_models_derived(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  m-awq:\n    provider: p1\n    logical: qwen14\n"
        "  m-q4:\n    provider: p2\n    logical: qwen14\n"
        "  solo:\n    provider: p3\n",
    )
    assert sorted(cfg.models.logical_models["qwen14"]) == ["m-awq", "m-q4"]
    assert "solo" not in cfg.models.logical_models.get("qwen14", [])


def test_flat_config_is_backward_compatible(tmp_path):
    cfg = _cfg(tmp_path, "models:\n  a:\n    provider: p\n")
    assert cfg.models.models["a"].logical is None
    assert cfg.models.models["a"].quant is None
    assert cfg.models.logical_models == {}


# --- Task 2: per-host exclusivity (derived) + query API --------------------

def test_exclusivity_derived_from_shared_provider_port(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  big-a:\n    provider: tray\n    port: 18400\n"
        "  big-b:\n    provider: tray\n    port: 18400\n"
        "  solo:\n    provider: tray\n    port: 9000\n",
    )
    assert cfg.models.exclusive_with("big-a") == ["big-b"]
    assert cfg.models.exclusive_with("big-b") == ["big-a"]
    assert cfg.models.exclusive_with("solo") == []


def test_cloud_no_port_is_not_exclusive(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n  c1:\n    provider: cloud\n  c2:\n    provider: cloud\n",
    )
    assert cfg.models.exclusive_with("c1") == []


def test_same_port_different_provider_is_not_exclusive(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  a:\n    provider: h1\n    port: 8000\n"
        "  b:\n    provider: h2\n    port: 8000\n",
    )
    assert cfg.models.exclusive_with("a") == []


def test_variants_and_logical_queries(tmp_path):
    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  v1:\n    provider: p\n    logical: L\n"
        "  v2:\n    provider: p2\n    logical: L\n",
    )
    assert cfg.models.variants_of("L") == ["v1", "v2"]
    assert cfg.models.logical_of("v1") == "L"
    assert cfg.models.logical_of("nope") is None
    assert cfg.models.variants_of("X") == []


# --- Task 3: introspection in /v1/available-models -------------------------

def test_available_payload_exposes_variants_and_exclusivity(tmp_path):
    from llm_relay.api.app import _build_available_payload
    from llm_relay.discovery.manager import DiscoveryManager

    cfg = _cfg(
        tmp_path,
        "models:\n"
        "  v1:\n    provider: tray\n    port: 18400\n    logical: big\n    quant: fp8\n"
        "  v2:\n    provider: tray\n    port: 18400\n    logical: big\n    quant: fp8-b\n"
        "  solo:\n    provider: p\n    port: 8000\n",
    )
    payload = _build_available_payload(cfg, DiscoveryManager())
    assert payload["logical_models"]["big"] == ["v1", "v2"]
    assert ["v1", "v2"] in payload["exclusivity_groups"]
    assert payload["v1"]["logical"] == "big"
    assert payload["v1"]["quant"] == "fp8"
    assert "logical" not in payload["solo"]
