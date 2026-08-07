"""Optional per-model description surfaced through the models endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from llm_relay.api.app import create_app


def _write_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text(
        "providers:\n  p:\n    base_url: http://127.0.0.1\n    ownership: ciq_owned\n"
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  fast-model:\n"
        "    provider: p\n"
        "    port: 8000\n"
        "    description: fast small model for bulk work\n"
        "  plain-model:\n"
        "    provider: p\n"
        "    port: 8001\n"
    )


def test_description_in_available_models(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    data = c.get("/v1/available-models").json()
    assert data["fast-model"]["description"] == "fast small model for bulk work"
    assert "description" not in data["plain-model"]


def test_description_in_openai_models_list(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    entries = {e["id"]: e for e in c.get("/v1/models").json()["data"]}
    described = [e for e in entries.values() if e.get("description")]
    assert any(e["description"] == "fast small model for bulk work" for e in described)


def test_description_in_model_card(tmp_path):
    _write_cfg(tmp_path)
    c = TestClient(create_app(config_dir=tmp_path))
    card = c.get("/v1/models/fast-model").json()
    assert card["description"] == "fast small model for bulk work"
