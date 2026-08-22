"""Admin-only prompt endpoints, and the trusted-listener bypass they close.

The auth middleware grants ``admin``+``cloud``+``third_party`` to ANY request
arriving on a trusted loopback port with no key at all. That is a deliberate
choice for operational state, and the wrong one for a searchable archive of
coworkers' conversations retained indefinitely. These tests pin the exception:
prompt routes demand a *presented* admin key even on the trusted listener, so
an on-box process cannot read conversation content keyless.

They also pin the audit trail. A search's query text is often more revealing of
intent than the rows it returns, so both the caller and the query are recorded.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from llm_relay.api.app import create_app
from llm_relay.api.middleware import AuthMiddleware
from llm_relay.auth import hash_key
from llm_relay.prompt_store import PromptStore, reset_store_for_tests

ADMIN = {"Authorization": "Bearer llmr_admin"}
PLAIN = {"Authorization": "Bearer llmr_plain"}

TRUSTED_PORT = 8090
AUTH_PORT = 8091

CONTENT_ROUTES = (
    "/admin/prompts/search?q=warewulf",
    "/admin/prompts/request/req-1",
    "/admin/prompts/stats",
)


def _keys() -> str:
    return (
        "keys:\n"
        f"  {hash_key('llmr_plain')}:\n    id: jdoe\n    priority_weight: 0.5\n"
        f"  {hash_key('llmr_admin')}:\n    id: brady\n    scopes: [admin]\n"
    )


@pytest.fixture()
def relay(tmp_path, monkeypatch):
    """A two-listener relay with capture OFF; tests opt in via ``_seed``."""
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    monkeypatch.delenv("LLM_RELAY_PROMPT_DB", raising=False)
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "audit.log"))
    (tmp_path / "auth.yaml").write_text(
        "auth:\n  enabled: true\n  trusted_ports: [%d]\n" % TRUSTED_PORT
    )
    (tmp_path / "api_keys.yaml").write_text(_keys())
    reset_store_for_tests()
    # No TestClient context manager anywhere in this module: entering it starts
    # the discovery pollers, which none of these read-path tests need.
    yield create_app(config_dir=tmp_path)
    reset_store_for_tests()


def _client(app, port=AUTH_PORT):
    return TestClient(app, base_url=f"http://testserver:{port}")


def _seed(tmp_path, monkeypatch):
    """Point capture at a fresh db and put one known request in it."""
    db = tmp_path / "prompts.db"
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", str(db))
    store = PromptStore(str(db))
    try:
        store.record({
            "request_id": "req-1",
            "ts": 1787000000.0,
            "day": "2026-08-20",
            "principal": "jdoe",
            "client": "claude-code",
            "model": "glm-5.2",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "how do I register a node in warewulf"},
            ],
            "completion": "use wwctl node add",
            "reasoning": "",
        })
        store.flush()
    finally:
        store.close()
    return db


def _audit_events(tmp_path):
    path = tmp_path / "audit.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# the middleware records HOW a caller authenticated
# --------------------------------------------------------------------------- #

def _auth_source(app, port, headers=()):
    """Run the middleware over a synthetic scope and return the state it set."""
    captured: dict = {}

    async def _inner(scope, receive, send):
        captured.update(scope.get("state") or {})

    scope = {
        "type": "http",
        "app": app,
        "server": ("127.0.0.1", port),
        "path": "/status",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers],
    }
    asyncio.run(AuthMiddleware(_inner)(scope, None, None))
    return captured.get("auth_source")


def test_auth_source_is_trusted_listener_on_a_trusted_port(relay):
    assert _auth_source(relay, TRUSTED_PORT) == "trusted_listener"


def test_auth_source_is_api_key_when_a_key_authenticated(relay):
    source = _auth_source(
        relay, AUTH_PORT, [("authorization", "Bearer llmr_admin")])
    assert source == "api_key"


def test_auth_source_is_auth_disabled_when_auth_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    (tmp_path / "auth.yaml").write_text("auth:\n  enabled: false\n")
    app = create_app(config_dir=tmp_path)
    assert _auth_source(app, AUTH_PORT) == "auth_disabled"


# --------------------------------------------------------------------------- #
# the guard: a keyless trusted-listener caller gets no content
# --------------------------------------------------------------------------- #

def test_trusted_listener_holds_admin_scope_on_other_admin_routes(relay):
    """Baseline for the test below: the bypass really is wide open elsewhere."""
    c = _client(relay, TRUSTED_PORT)
    assert c.get("/logs").status_code == 200
    # 404, not 401/403: the route ran with admin scope and no key at all.
    assert c.post("/admin/pause", json={"provider": "nope"}).status_code == 404


def test_trusted_listener_is_refused_prompt_content(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = _client(relay, TRUSTED_PORT)
    for route in CONTENT_ROUTES:
        r = c.get(route)
        assert r.status_code == 403, route
        assert "admin API key" in r.json()["detail"], route
        body = r.text.lower()
        assert "warewulf" not in body, route
        assert "wwctl" not in body, route


def test_trusted_listener_refusal_is_audited(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    _client(relay, TRUSTED_PORT).get("/admin/prompts/request/req-1")
    denied = [e for e in _audit_events(tmp_path)
              if e["event"] == "prompt_access_denied"]
    assert denied, "a keyless attempt to read the archive must leave a record"
    assert denied[-1]["auth_source"] == "trusted_listener"
    assert denied[-1]["path"] == "/admin/prompts/request/req-1"


def test_an_exempt_path_is_still_refused(tmp_path, monkeypatch):
    """A missing ``auth_source`` must read as untrusted, not as a default.

    The exempt-path branch skips authentication entirely and sets no
    ``auth_source``. Nothing today puts a prompt route in ``exempt_paths``, and
    this test is what keeps a future edit there from quietly opening the
    archive.
    """
    monkeypatch.delenv("LLM_RELAY_AUTH", raising=False)
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "audit.log"))
    (tmp_path / "auth.yaml").write_text(
        "auth:\n  enabled: true\n  trusted_ports: []\n"
        "  exempt_paths: ['/health', '/admin/prompts/stats']\n"
    )
    (tmp_path / "api_keys.yaml").write_text(_keys())
    reset_store_for_tests()
    try:
        _seed(tmp_path, monkeypatch)
        app = create_app(config_dir=tmp_path)
        r = _client(app).get("/admin/prompts/stats")
        assert r.status_code == 403
        assert "admin API key" in r.json()["detail"]
    finally:
        reset_store_for_tests()


def test_non_admin_key_is_refused_by_the_middleware(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = _client(relay)
    for route in CONTENT_ROUTES:
        r = c.get(route, headers=PLAIN)
        assert r.status_code == 403, route
        assert r.json()["detail"] == "admin scope required", route


# --------------------------------------------------------------------------- #
# a real admin key gets through
# --------------------------------------------------------------------------- #

def test_real_admin_key_reads_stats(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/stats", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["requests"] == 1
    assert body["stored_messages"] == 2
    assert body["codec"] in ("zlib", "zstd")


def test_search_returns_matching_rows(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/search?q=warewulf", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["count"] == 1
    hit = body["hits"][0]
    assert hit["request_id"] == "req-1"
    assert hit["role"] == "user"
    assert hit["principal"] == "jdoe"
    assert "warewulf" in hit["snippet"]


def test_search_filters_narrow_the_result(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = _client(relay)
    assert c.get("/admin/prompts/search?q=warewulf&role=system",
                 headers=ADMIN).json()["count"] == 0
    assert c.get("/admin/prompts/search?q=warewulf&principal=someone-else",
                 headers=ADMIN).json()["count"] == 0
    assert c.get("/admin/prompts/search?q=warewulf&model=glm-5.2",
                 headers=ADMIN).json()["count"] == 1


def test_search_limit_is_capped(relay, tmp_path, monkeypatch):
    """An unbounded limit on a content route is an exfiltration convenience."""
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/search?q=warewulf&limit=100000",
                           headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["limit"] == 200


def test_search_with_no_usable_terms_returns_no_rows(relay, tmp_path, monkeypatch):
    """FTS metacharacters must not become a 500 on an admin route."""
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/search?q=NEAR(%22", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["hits"] == []


def test_read_request_returns_messages(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/request/req-1", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["found"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "how do I register a node in warewulf"
    assert body["completion"] == "use wwctl node add"


def test_read_of_an_unknown_request_is_not_found(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client(relay).get("/admin/prompts/request/nope", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["found"] is False
    assert r.json()["messages"] == []


# --------------------------------------------------------------------------- #
# auditing
# --------------------------------------------------------------------------- #

def test_search_audits_the_caller_and_the_query_text(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    _client(relay).get("/admin/prompts/search?q=warewulf&principal=jdoe",
                       headers=ADMIN)
    events = [e for e in _audit_events(tmp_path) if e["event"] == "prompt_search"]
    assert len(events) == 1
    assert events[0]["by"] == "brady"
    assert events[0]["query"] == "warewulf"
    assert events[0]["principal"] == "jdoe"
    assert events[0]["results"] == 1


def test_read_audits_the_caller_and_the_request_id(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    _client(relay).get("/admin/prompts/request/req-1", headers=ADMIN)
    events = [e for e in _audit_events(tmp_path) if e["event"] == "prompt_read"]
    assert len(events) == 1
    assert events[0]["by"] == "brady"
    assert events[0]["request_id"] == "req-1"
    assert events[0]["found"] is True


def test_stats_is_audited_too(relay, tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    _client(relay).get("/admin/prompts/stats", headers=ADMIN)
    events = [e for e in _audit_events(tmp_path) if e["event"] == "prompt_stats"]
    assert len(events) == 1
    assert events[0]["by"] == "brady"


def test_a_search_is_audited_even_when_capture_is_off(relay, tmp_path):
    """Turning capture off must not silently stop the record of who looked."""
    _client(relay).get("/admin/prompts/search?q=warewulf", headers=ADMIN)
    events = [e for e in _audit_events(tmp_path) if e["event"] == "prompt_search"]
    assert len(events) == 1
    assert events[0]["query"] == "warewulf"
    assert events[0]["enabled"] is False


# --------------------------------------------------------------------------- #
# capture off
# --------------------------------------------------------------------------- #

def test_endpoints_report_disabled_when_no_prompt_db(relay):
    c = _client(relay)
    search = c.get("/admin/prompts/search?q=warewulf", headers=ADMIN)
    assert search.status_code == 200
    assert search.json() == {"enabled": False, "query": "warewulf",
                             "hits": [], "count": 0, "limit": 50}

    read = c.get("/admin/prompts/request/req-1", headers=ADMIN)
    assert read.status_code == 200
    assert read.json() == {"enabled": False, "request_id": "req-1",
                           "found": False, "messages": []}

    stats = c.get("/admin/prompts/stats", headers=ADMIN)
    assert stats.status_code == 200
    assert stats.json()["enabled"] is False
