"""The usage read API is admin-scoped and shaped as the portal expects."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_relay import usage_store


def _min_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    return tmp_path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "usage.db")
    monkeypatch.setenv("LLM_RELAY_USAGE_DB", db)
    usage_store.reset_store_for_tests()
    conn = usage_store.open_db(db)
    conn.execute(
        "INSERT INTO requests (request_id, ts, day, principal, client, model, "
        "provider, outcome, streamed, input_tokens, output_tokens, "
        "reasoning_tokens, usage_source, reasoning_source) VALUES "
        "('r1', 100.0, '2026-08-20', 'brady', 'claude-code', 'glm-5.2', 'gb200', "
        "'success', 1, 5000, 400, 120, 'upstream_final', 'upstream_details')"
    )
    conn.close()

    from llm_relay.api.app import create_app

    app = create_app(config_dir=_min_cfg(tmp_path))
    yield TestClient(app)
    usage_store.reset_store_for_tests()


def test_rollup_returns_rows(client):
    r = client.get("/admin/usage/rollup?start=2026-08-01&end=2026-08-31")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert rows[0]["input_tokens"] == 5000
    assert rows[0]["reasoning_tokens"] == 120


def test_summary_returns_per_principal_totals(client):
    r = client.get("/admin/usage/summary")
    assert r.status_code == 200
    assert r.json()["by_principal"]["brady"]["all_time_input_tokens"] == 5000


def test_summary_reports_a_real_event_timestamp(client):
    """The column exists so "last active" stops being a scrape time."""
    r = client.get("/admin/usage/summary")
    assert r.json()["by_principal"]["brady"]["last_activity_ts"] == 100.0


def test_rollup_rejects_a_malformed_date(client):
    r = client.get("/admin/usage/rollup?start=not-a-date&end=2026-08-31")
    assert r.status_code == 400


def test_health_reports_store_size(client):
    r = client.get("/admin/usage/health")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 1
    assert body["bytes"] > 0


def test_endpoints_report_disabled_when_the_store_is_unconfigured(tmp_path, monkeypatch):
    """Unset ``LLM_RELAY_USAGE_DB`` is the shipping default, so the read API has
    to answer empty rather than 500."""
    monkeypatch.delenv("LLM_RELAY_USAGE_DB", raising=False)
    usage_store.reset_store_for_tests()
    from llm_relay.api.app import create_app

    c = TestClient(create_app(config_dir=_min_cfg(tmp_path)))
    assert c.get("/admin/usage/rollup?start=2026-08-01&end=2026-08-31").json() == {
        "rows": [], "enabled": False,
    }
    assert c.get("/admin/usage/summary").json()["enabled"] is False
    assert c.get("/admin/usage/health").json()["rows"] == 0


def test_admin_usage_requires_the_admin_scope(tmp_path, monkeypatch):
    """The existing middleware gates /admin; assert it actually covers the new
    paths so a future route move cannot silently expose fleet-wide usage."""
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "audit.log"))
    from llm_relay.auth import hash_key

    cfg_dir = _min_cfg(tmp_path)
    (cfg_dir / "auth.yaml").write_text("auth:\n  enabled: true\n  trusted_ports: [8090]\n")
    (cfg_dir / "api_keys.yaml").write_text(
        "keys:\n"
        f"  {hash_key('llmr_plain')}:\n    id: jdoe\n"
        f"  {hash_key('llmr_admin')}:\n    id: brady\n    scopes: [admin]\n"
    )
    from llm_relay.api.app import create_app

    app = create_app(config_dir=cfg_dir)
    c = TestClient(app, base_url="http://testserver:8091")
    q = "/admin/usage/rollup?start=2026-08-01&end=2026-08-31"
    assert c.get(q).status_code == 401
    assert c.get(q, headers={"Authorization": "Bearer llmr_plain"}).status_code == 403
    assert c.get(q, headers={"Authorization": "Bearer llmr_admin"}).status_code == 200
