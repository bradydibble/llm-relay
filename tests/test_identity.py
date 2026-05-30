"""Host-qualified model identity helpers (provider:model)."""
from __future__ import annotations

from llm_relay.config.types import ModelConfig
from llm_relay.routing.keys import compose_model_id, resolve_model_id


def _models() -> dict[str, ModelConfig]:
    # Same base model name on two providers — the case bare names can't express.
    return {
        "model-x": ModelConfig(provider="prov-a"),
        "model-y": ModelConfig(provider="prov-b"),
        "shared": ModelConfig(provider="prov-a"),
    }


def test_compose_model_id_joins_provider_and_model():
    assert compose_model_id("prov-a", "model-x") == "prov-a:model-x"


def test_resolve_model_id_accepts_bare_name():
    assert resolve_model_id(_models(), "model-x") == "model-x"


def test_resolve_model_id_accepts_matching_qualified_id():
    assert resolve_model_id(_models(), "prov-a:model-x") == "model-x"


def test_resolve_model_id_rejects_mismatched_provider():
    # model-x is served by prov-a, so prov-b:model-x is not a valid pairing.
    assert resolve_model_id(_models(), "prov-b:model-x") is None


def test_resolve_model_id_unknown_returns_none():
    assert resolve_model_id(_models(), "prov-a:nope") is None
    assert resolve_model_id(_models(), "totally-unknown") is None


def test_resolve_model_id_qualified_round_trips_with_compose():
    models = _models()
    qid = compose_model_id("prov-a", "shared")
    assert qid == "prov-a:shared"
    assert resolve_model_id(models, qid) == "shared"
