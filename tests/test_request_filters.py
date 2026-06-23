"""Per-model request filters (plan 5): strip/set params before forwarding."""
from __future__ import annotations

from llm_relay.config.loader import ConfigLoader
from llm_relay.discovery.manager import DiscoveryManager
from llm_relay.routing.router import RequestRouter


def _router(tmp_path, models_yaml: str, providers: str = "providers: {}\n") -> RequestRouter:
    (tmp_path / "providers.yaml").write_text(providers)
    (tmp_path / "models.yaml").write_text(models_yaml)
    c = ConfigLoader(config_dir=tmp_path)
    c.load()
    return RequestRouter(c, DiscoveryManager())


def test_strip_params_removes_keys(tmp_path):
    r = _router(tmp_path, "models:\n  m:\n    provider: p\n    strip_params: [logprobs, foo]\n")
    out = r._apply_filters({"messages": [], "logprobs": True, "foo": 1, "temperature": 0.5}, "m")
    assert "logprobs" not in out and "foo" not in out
    assert out["temperature"] == 0.5


def test_set_params_overrides(tmp_path):
    r = _router(tmp_path, "models:\n  m:\n    provider: p\n    set_params: {temperature: 0.0}\n")
    out = r._apply_filters({"messages": [], "temperature": 0.9}, "m")
    assert out["temperature"] == 0.0


def test_no_filters_returns_input_unchanged(tmp_path):
    r = _router(tmp_path, "models:\n  m:\n    provider: p\n")
    body = {"messages": [], "temperature": 0.9}
    assert r._apply_filters(body, "m") == body


def test_filters_do_not_mutate_caller_dict(tmp_path):
    r = _router(tmp_path, "models:\n  m:\n    provider: p\n    set_params: {temperature: 0.0}\n")
    body = {"messages": [], "temperature": 0.9}
    r._apply_filters(body, "m")
    assert body["temperature"] == 0.9


def test_unknown_model_returns_input_unchanged(tmp_path):
    r = _router(tmp_path, "models:\n  m:\n    provider: p\n")
    body = {"messages": []}
    assert r._apply_filters(body, "nonexistent") == body
