"""Jobs are principal-scoped on the auth listener."""
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
        f"  {hash_key('llmr_owner')}:\n    id: jdoe\n"
        f"  {hash_key('llmr_other')}:\n    id: msmith\n"
        f"  {hash_key('llmr_admin')}:\n    id: brady\n    scopes: [admin]\n"
    )
    return create_app(config_dir=tmp_path)


def _client(app, port):
    return TestClient(app, base_url=f"http://testserver:{port}")


def _submit(c, key):
    r = c.post(
        "/v1/jobs",
        json={"model": "main", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 202
    return r.json()["job_id"]


def test_owner_sees_own_job(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    jid = _submit(c, "llmr_owner")
    r = c.get(f"/v1/jobs/{jid}", headers={"Authorization": "Bearer llmr_owner"})
    assert r.status_code == 200
    assert r.json()["principal"] == "jdoe"


def test_other_principal_gets_404(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    jid = _submit(c, "llmr_owner")
    assert c.get(f"/v1/jobs/{jid}", headers={"Authorization": "Bearer llmr_other"}).status_code == 404
    assert c.post(
        f"/v1/jobs/{jid}/cancel", headers={"Authorization": "Bearer llmr_other"}
    ).status_code == 404


def test_admin_sees_all_jobs(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    jid = _submit(c, "llmr_owner")
    assert c.get(f"/v1/jobs/{jid}", headers={"Authorization": "Bearer llmr_admin"}).status_code == 200


def test_trusted_listener_sees_all_jobs(app_two_listeners):
    auth_c = _client(app_two_listeners, 8091)
    jid = _submit(auth_c, "llmr_owner")
    trusted = _client(app_two_listeners, 8090)
    assert trusted.get(f"/v1/jobs/{jid}").status_code == 200
