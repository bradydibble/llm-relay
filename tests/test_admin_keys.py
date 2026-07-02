"""Key lifecycle over HTTP: /admin/keys mint/list/revoke with live reload."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_relay.api.app import create_app
from llm_relay.auth import hash_key


@pytest.fixture()
def app_two_listeners(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "audit.log"))
    (tmp_path / "auth.yaml").write_text(
        "auth:\n  enabled: true\n  trusted_ports: [8090]\n"
    )
    (tmp_path / "api_keys.yaml").write_text(
        "keys:\n"
        f"  {hash_key('llmr_plain')}:\n    id: jdoe\n    priority_weight: 0.5\n"
        f"  {hash_key('llmr_admin')}:\n    id: brady\n    scopes: [admin]\n"
    )
    return create_app(config_dir=tmp_path)


def _client(app, port):
    return TestClient(app, base_url=f"http://testserver:{port}")


ADMIN = {"Authorization": "Bearer llmr_admin"}


def test_admin_can_mint_and_new_key_works(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    r = c.post("/admin/keys", json={"id": "newguy", "priority_weight": 0.5}, headers=ADMIN)
    assert r.status_code == 200
    key = r.json()["key"]
    assert key.startswith("llmr_")
    ok = c.get("/status", headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200


def test_admin_mint_refuses_scopes(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    r = c.post("/admin/keys", json={"id": "evil", "scopes": ["admin"]}, headers=ADMIN)
    assert r.status_code == 400


def test_admin_list_and_revoke(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    key = c.post("/admin/keys", json={"id": "victim"}, headers=ADMIN).json()["key"]
    listing = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    target = next(k for k in listing if k["id"] == "victim")
    assert target["created"][:2] == "20"
    r = c.delete(f"/admin/keys/{target['hash_prefix']}", headers=ADMIN)
    assert r.status_code == 200
    dead = c.get("/status", headers={"Authorization": f"Bearer {key}"})
    assert dead.status_code == 401


def test_admin_keys_denied_without_admin_scope(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    r = c.get("/admin/keys", headers={"Authorization": "Bearer llmr_plain"})
    assert r.status_code == 403


def test_revoke_unknown_prefix_404(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    assert c.delete("/admin/keys/ffffffffffff", headers=ADMIN).status_code == 404
