"""Naming a model by exact id is a requirement, not a hint.

When a caller pins a specific model and it is not serving, the relay must say so
plainly and terminally. It must never substitute a different model (that is the
silent-degradation failure this work exists to remove), and it must not dress the
outage up as retryable backpressure — the relay cannot tell whether a pinned
backend is five seconds from returning or has been down for a week.

Aliases are the deliberate counterpart: they ARE open over the live fleet, so
their members returning is the genuine remedy and the Retry-After path stays.
"""
import pytest
import yaml
from fastapi import HTTPException

from llm_relay.api.app import create_app
from llm_relay.config.types import EndpointState, EndpointStatus, NoBackendAvailableError
from llm_relay.discovery.endpoint import EndpointClient
from llm_relay.routing.selector import RoutingContext

PROVIDERS = {
    "providers": {
        "local-llm": {
            "type": "openai",
            "base_url": "http://127.0.0.1",
            "ownership": "ciq_owned",
            "enabled": True,
        }
    }
}
MODELS = {
    "models": {
        "pinned-model": {"provider": "local-llm", "port": 8080, "preference": 0.9,
                         "use_cases": {"main": 1}},
        "other-model": {"provider": "local-llm", "port": 8081, "preference": 0.5,
                        "use_cases": {"main": 2}},
    }
}


def _app(tmp_path, live: list[str] | None = None):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "providers.yaml").write_text(yaml.safe_dump(PROVIDERS))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(MODELS))
    app = create_app(config_dir=cfg_dir)
    for m in live or []:
        state = EndpointState(provider="local-llm", status=EndpointStatus.healthy, models=[m])
        app.state.discovery.clients[f"k::{m}"] = EndpointClient(
            provider_name="local-llm", base_url="http://127.0.0.1", state=state
        )
        app.state.discovery.model_to_client[m] = f"k::{m}"
    return app


def _body(model: str):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


async def test_named_dead_model_is_terminal_and_names_itself(tmp_path):
    """other-model is live and would happily serve — the point is that it MUST NOT,
    and that the error names the model the caller actually asked for."""
    router = _app(tmp_path, live=["other-model"]).state.router
    with pytest.raises(HTTPException) as exc:
        await router.route_and_forward(_body("pinned-model"))
    assert exc.value.status_code == 503
    assert exc.value.headers["X-Llm-Relay-Error"] == "named_model_unavailable"
    detail = exc.value.detail
    assert "pinned-model" in detail["error"]
    assert detail["named_model"]["model"] == "pinned-model"
    assert "does not substitute" in detail["named_model"]["remedy"]


async def test_named_dead_model_carries_no_retry_after(tmp_path):
    """It must be an HTTPException, not NoBackendAvailableError — the latter is
    what the API layer turns into a Retry-After."""
    router = _app(tmp_path, live=["other-model"]).state.router
    with pytest.raises(HTTPException):
        await router.route_and_forward(_body("pinned-model"))


async def test_alias_with_dead_members_keeps_retryable_backpressure(tmp_path):
    """Regression guard: aliases are open over the fleet, so a member returning is
    the real remedy. Batch callers depend on this Retry-After."""
    router = _app(tmp_path, live=[]).state.router  # nothing live at all
    with pytest.raises(NoBackendAvailableError):
        await router.route_and_forward(_body("main"))


async def test_live_named_model_still_routes(tmp_path):
    """Sanity: the loud-fail must not break the ordinary pinned-and-healthy path."""
    router = _app(tmp_path, live=["pinned-model"]).state.router
    ctx = RoutingContext(requested_model="pinned-model")
    assert router.selector.select_best(ctx) == "pinned-model"


# --- explicit_target: what counts as "named" -------------------------------

def test_explicit_target_identifies_a_concrete_model(tmp_path):
    sel = _app(tmp_path).state.router.selector
    assert sel.explicit_target(RoutingContext(requested_model="pinned-model")) == "pinned-model"


def test_explicit_target_resolves_host_qualified_id(tmp_path):
    """'local-llm:pinned-model' names the same concrete model."""
    sel = _app(tmp_path).state.router.selector
    got = sel.explicit_target(RoutingContext(requested_model="local-llm:pinned-model"))
    assert got == "pinned-model"


def test_explicit_target_is_none_for_alias(tmp_path):
    sel = _app(tmp_path).state.router.selector
    assert sel.explicit_target(RoutingContext(requested_model="main")) is None


def test_explicit_target_is_none_for_unknown_id(tmp_path):
    """An unknown id is best-effort fleet routing, not a pin — it keeps the
    open-fallthrough behaviour and must not become a hard failure."""
    sel = _app(tmp_path).state.router.selector
    assert sel.explicit_target(RoutingContext(requested_model="gpt-9-turbo")) is None
