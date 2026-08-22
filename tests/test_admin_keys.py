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
        f"    owner_email: jdoe@example.com\n"
        f"  {hash_key('llmr_admin')}:\n    id: owner\n    owner_email: owner@example.com\n    scopes: [admin]\n"
        f"  {hash_key('llmr_portal')}:\n    id: svc-portal\n    scopes: [key_issuer]\n"
    )
    return create_app(config_dir=tmp_path)


def _client(app, port):
    return TestClient(app, base_url=f"http://testserver:{port}")


ADMIN = {"Authorization": "Bearer llmr_admin"}
PORTAL = {"Authorization": "Bearer llmr_portal"}


def test_key_issuer_lists_only_the_requested_owner_records(app_two_listeners):
    """A service lacking the owner-key route must never expose the key store.

    The portal will supply this address only after deriving it from its Okta
    identity header. The relay returns record metadata, never a bearer value.
    """
    c = _client(app_two_listeners, 8091)
    r = c.post("/portal/owner-keys/list", json={"owner_email": "jdoe@example.com"}, headers=PORTAL)
    assert r.status_code == 200
    assert r.json() == {"keys": [{
        "hash_prefix": hash_key("llmr_plain")[:12],
        "id": "jdoe",
        "owner_email": "jdoe@example.com",
        "scopes": [],
        "priority_weight": 0.5,
        "enabled": True,
        "created": "",
        "note": "",
    }]}


def test_key_issuer_mints_and_revokes_only_its_owners_non_admin_token(app_two_listeners):
    """A user-provisioned key is usable, scoped by default, and self-revocable."""
    c = _client(app_two_listeners, 8091)
    minted = c.post("/portal/owner-keys", json={"owner_email": "jdoe@example.com"}, headers=PORTAL)
    assert minted.status_code == 200
    body = minted.json()
    key = body["key"]
    assert key.startswith("llmr_")
    assert body["id"] == "jdoe"
    assert body["scopes"] == ["cloud", "third_party"]
    assert c.get("/status", headers={"Authorization": f"Bearer {key}"}).status_code == 200

    listed = c.post("/portal/owner-keys/list", json={"owner_email": "jdoe@example.com"}, headers=PORTAL)
    issued = next(record for record in listed.json()["keys"] if record["created"])
    assert issued["scopes"] == ["cloud", "third_party"]
    assert "key" not in issued

    revoked = c.request(
        "DELETE", f"/portal/owner-keys/{issued['hash_prefix']}",
        json={"owner_email": "jdoe@example.com"}, headers=PORTAL,
    )
    assert revoked.status_code == 200 and revoked.json() == {"revoked": 1}
    assert c.get("/status", headers={"Authorization": f"Bearer {key}"}).status_code == 401


def test_admin_key_listing_includes_owner_email_without_bearer_value(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    listing = c.get("/admin/keys", headers=ADMIN)
    assert listing.status_code == 200
    record = next(item for item in listing.json()["keys"] if item["id"] == "jdoe")
    assert record["owner_email"] == "jdoe@example.com"
    assert "key" not in record


def test_admin_mint_defaults_to_cloud_and_third_party_for_an_owner(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    minted = c.post(
        "/admin/keys",
        json={"id": "newuser", "owner_email": "newuser@example.com"}, headers=ADMIN,
    )
    assert minted.status_code == 200
    listing = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    record = next(item for item in listing if item["id"] == "newuser")
    assert record["owner_email"] == "newuser@example.com"
    assert record["scopes"] == ["cloud", "third_party"]


def test_owner_key_routes_require_key_issuer_and_protect_admin_keys(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    forbidden = c.post(
        "/portal/owner-keys/list", json={"owner_email": "jdoe@example.com"},
        headers={"Authorization": "Bearer llmr_plain"},
    )
    assert forbidden.status_code == 403

    revoke_admin = c.request(
        "DELETE", f"/portal/owner-keys/{hash_key('llmr_admin')[:12]}",
        json={"owner_email": "owner@example.com"}, headers=PORTAL,
    )
    assert revoke_admin.status_code == 403


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


def test_admin_can_patch_scopes(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    c.post("/admin/keys", json={"id": "patchable"}, headers=ADMIN)
    listing = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    target = next(k for k in listing if k["id"] == "patchable")
    r = c.patch(f"/admin/keys/{target['hash_prefix']}",
                json={"scopes": ["third_party"]}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["scopes"] == ["third_party"]
    updated = next(k for k in c.get("/admin/keys", headers=ADMIN).json()["keys"]
                   if k["id"] == "patchable")
    assert updated["scopes"] == ["third_party"]


def test_admin_patch_can_clear_scopes(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    c.post("/admin/keys", json={"id": "clearable"}, headers=ADMIN)
    listing = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    target = next(k for k in listing if k["id"] == "clearable")
    c.patch(f"/admin/keys/{target['hash_prefix']}", json={"scopes": ["third_party"]}, headers=ADMIN)
    r = c.patch(f"/admin/keys/{target['hash_prefix']}", json={"scopes": []}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["scopes"] == []


def test_admin_patch_refuses_admin_scope(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    c.post("/admin/keys", json={"id": "target"}, headers=ADMIN)
    listing = c.get("/admin/keys", headers=ADMIN).json()["keys"]
    target = next(k for k in listing if k["id"] == "target")
    r = c.patch(f"/admin/keys/{target['hash_prefix']}", json={"scopes": ["admin"]}, headers=ADMIN)
    assert r.status_code == 400


def test_patch_unknown_prefix_404(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    assert c.patch("/admin/keys/ffffffffffff", json={"scopes": []}, headers=ADMIN).status_code == 404


def test_patch_denied_without_admin_scope(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    assert c.patch("/admin/keys/abc", json={"scopes": []},
                   headers={"Authorization": "Bearer llmr_plain"}).status_code == 403
