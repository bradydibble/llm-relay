"""/routing-table/{name} resolves aliases, not just concrete models."""
from __future__ import annotations

from fastapi.testclient import TestClient

from llm_relay.api.app import create_app


def _write_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text(
        "providers:\n  p:\n    base_url: http://127.0.0.1\n"
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  fast-model:\n"
        "    provider: p\n"
        "    port: 8000\n"
        "    use_cases: {main: 2}\n"
        "  slow-model:\n"
        "    provider: p\n"
        "    port: 8001\n"
        "    use_cases: {main: 1}\n"
    )


def test_routing_table_resolves_alias(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    r = c.get("/routing-table/main")
    assert r.status_code == 200
    body = r.json()
    assert body["alias"] == "main"
    assert body["members"] == ["fast-model", "slow-model"]
    # no discovery in this test -> nothing available -> resolved is None
    assert body["resolved"] is None


def test_routing_table_concrete_model_unchanged(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    r = c.get("/routing-table/fast-model")
    assert r.status_code == 200
    assert r.json()["model"] == "fast-model"


def test_routing_table_unknown_404(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    assert c.get("/routing-table/nope").status_code == 404
