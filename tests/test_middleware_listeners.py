"""Listener-aware auth: trusted-port bypass, auth-port enforcement, scope gate."""
from __future__ import annotations

import re

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


def _auth_failure_count() -> float:
    from llm_relay import metrics

    body, _ = metrics.render_exposition()
    m = re.search(rb"^llm_relay_auth_failures_total ([0-9.]+)$", body, re.M)
    return float(m.group(1)) if m else 0.0


def test_trusted_port_needs_no_key(app_two_listeners):
    r = _client(app_two_listeners, 8090).get("/status")
    assert r.status_code == 200


def test_auth_port_401_without_key(app_two_listeners):
    r = _client(app_two_listeners, 8091).get("/status")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_auth_port_ok_with_key(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    r = c.get("/status", headers={"Authorization": "Bearer llmr_plain"})
    assert r.status_code == 200


def test_health_exempt_on_auth_port(app_two_listeners):
    r = _client(app_two_listeners, 8091).get("/health")
    assert r.status_code == 200
    # minimal body when auth is on: no backend topology for keyless callers
    assert r.json() == {"status": "ok"}


def test_metrics_requires_key_on_auth_port(app_two_listeners):
    assert _client(app_two_listeners, 8091).get("/metrics").status_code == 401


def test_logs_needs_admin_scope(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    assert c.get("/logs", headers={"Authorization": "Bearer llmr_plain"}).status_code == 403
    assert c.get("/logs", headers={"Authorization": "Bearer llmr_admin"}).status_code == 200


def test_admin_needs_admin_scope(app_two_listeners):
    c = _client(app_two_listeners, 8091)
    r = c.post(
        "/admin/pause",
        json={"provider": "nope"},
        headers={"Authorization": "Bearer llmr_plain"},
    )
    assert r.status_code == 403


def test_trusted_port_reaches_admin_and_logs(app_two_listeners):
    c = _client(app_two_listeners, 8090)
    assert c.get("/logs").status_code == 200
    # unknown provider -> 404 proves the route ran (not 401/403)
    assert c.post("/admin/pause", json={"provider": "nope"}).status_code == 404


def test_auth_failure_metric_increments(app_two_listeners):
    before = _auth_failure_count()
    _client(app_two_listeners, 8091).get("/status")  # 401
    assert _auth_failure_count() >= before + 1


def test_audit_log_records_auth_failure(app_two_listeners, tmp_path):
    _client(app_two_listeners, 8091).get("/status")  # 401
    lines = (tmp_path / "audit.log").read_text().strip().splitlines()
    assert any('"event": "auth_failure"' in line for line in lines)
