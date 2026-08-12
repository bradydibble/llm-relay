"""Capability claims in /v1/models are routable truth, not decoration.

Two invariants, both bought with real incidents (2026-08-11):

1. The OpenAI-shaped model list carries a ``capabilities`` block and a
   ``limit.context``, because that is the schema third-party pickers (and our
   own ciq-harness client) actually read. Publishing context only as
   ``context_length`` left our client filling in a 131072 default for every
   model — overstating the 32k models and halving the 262k ones — and a
   fabricated ``toolcall: true`` for a model that cannot emit a tool call.

2. A request that CARRIES tools requires a tool-capable backend, whether or not
   the client knew to send ``X-Llm-Relay-Require-Tools``. Header-only derivation
   let a tool-bearing request open-fallthrough onto qwen3-14b, which "succeeded"
   with ``finish_reason: stop`` and empty content — a failure the client cannot
   even detect.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from llm_relay.api.app import _build_model_card, _build_models_list_payload
from llm_relay.config.loader import ConfigLoader
from llm_relay.config.types import EndpointState, EndpointStatus
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.discovery.manager import DiscoveryManager


def _disc(*models: str) -> DiscoveryManager:
    """Discovery reporting each model healthy, same stub as test_confidentiality:
    selection and payload-building are then driven purely by config."""
    disc = DiscoveryManager()
    for m in models:
        state = EndpointState(provider="node-a", status=EndpointStatus.healthy, models=[m])
        disc.clients[f"k::{m}"] = EndpointClient(provider_name="node-a", base_url="x", state=state)
        disc.model_to_client[m] = f"k::{m}"
    return disc


def _cfg(tmp_path: Path) -> ConfigLoader:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump({
        "providers": {"node-a": {"type": "openai", "base_url": "http://127.0.0.1",
                                 "ownership": "ciq_owned", "enabled": True}}
    }))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump({
        "models": {
            # A full-featured reasoning model...
            "prose-tools": {"provider": "node-a", "class": "unknown", "privacy": "local_only",
                            "port": 8001, "context_window": 262144,
                            "capabilities": ["tool_use", "structured_output", "reasoning"],
                            "use_cases": {"mixed": 1}},
            # ...and one that genuinely cannot call tools (the qwen3-14b case).
            "prose-only": {"provider": "node-a", "class": "unknown", "privacy": "local_only",
                           "port": 8002, "context_window": 32768,
                           "capabilities": ["structured_output"],
                           "use_cases": {"mixed": 2}},
        },
    }))
    (cfg_dir / "policy.yaml").write_text(yaml.safe_dump(
        {"policy": {"fallback": {"retry_on": ["503"]}}}
    ))
    loader = ConfigLoader(cfg_dir)
    loader.load()
    return loader


def test_models_list_publishes_capabilities_and_limit(tmp_path):
    cfg = _cfg(tmp_path)
    payload = _build_models_list_payload(cfg, _disc("prose-tools", "prose-only"))
    by_id = {e["id"]: e for e in payload["data"]}

    full = by_id["node-a:prose-tools"]
    assert full["capabilities"] == {"toolcall": True, "reasoning": True,
                                    "structured_output": True}
    assert full["limit"]["context"] == 262144

    limited = by_id["node-a:prose-only"]
    assert limited["capabilities"]["toolcall"] is False, \
        "a model without tool_use must not be advertised as tool-capable"
    assert limited["limit"]["context"] == 32768, \
        "the real window, not a client-side default"


def test_alias_capabilities_are_the_intersection(tmp_path):
    """An alias only advertises what EVERY member delivers: the client cannot
    know which member answers, so a union claim is a lie some of the time."""
    cfg = _cfg(tmp_path)
    card = _build_model_card(cfg, _disc("prose-tools", "prose-only"), "mixed")
    assert card is not None
    assert card["capabilities"]["toolcall"] is False, \
        "one member lacks tool_use, so the alias must not claim it"
    assert card["capabilities"]["structured_output"] is True, \
        "every member has it, so the alias may claim it"


def test_tools_in_body_require_tool_capable_backend(tmp_path):
    """No header, tools in the body: the tool-less model must be filtered even
    though it is otherwise eligible (and higher priority here)."""
    from llm_relay.routing.selector import ModelSelector, RoutingContext

    cfg = _cfg(tmp_path)
    selector = ModelSelector(cfg, _disc("prose-tools", "prose-only"))
    ctx = RoutingContext(requested_model="mixed", require_tools=True)
    names = [c.model for c in selector.select_chain(ctx)]
    assert "prose-only" not in names, \
        "a tool-bearing request must never land on a model that cannot emit tool_calls"
