"""The usage read API is admin-scoped and shaped as the portal expects."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_relay import cache_sampler, usage_store


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


@pytest.fixture()
def latency_client(tmp_path, monkeypatch):
    """A store with real observations on one day and a synthetic backfill row on
    another, so the endpoint can be shown to percentile only what was measured.
    """
    db = str(tmp_path / "usage.db")
    monkeypatch.setenv("LLM_RELAY_USAGE_DB", db)
    usage_store.reset_store_for_tests()
    conn = usage_store.open_db(db)
    for i, (duration, ttft) in enumerate(((300, 30), (100, 10), (4000, 40), (200, 20))):
        conn.execute(
            "INSERT INTO requests (request_id, ts, day, principal, client, model, "
            "provider, outcome, streamed, duration_ms, ttft_ms, usage_source, "
            "synthetic) VALUES (?, 100.0, '2026-08-20', 'brady', 'claude-code', "
            "'glm-5.2', 'gb200', 'success', 1, ?, ?, 'upstream_final', 0)",
            (f"l{i}", duration, ttft),
        )
    conn.execute(
        "INSERT INTO requests (request_id, ts, day, principal, client, model, "
        "provider, outcome, streamed, duration_ms, ttft_ms, usage_source, "
        "synthetic, request_count) VALUES ('b1', 100.0, '2026-08-19', 'brady', "
        "'claude-code', 'glm-5.2', '', 'success', 0, 999999, 999999, "
        "'prom_backfill', 1, 3181)"
    )
    conn.close()

    from llm_relay.api.app import create_app

    app = create_app(config_dir=_min_cfg(tmp_path))
    yield TestClient(app)
    usage_store.reset_store_for_tests()


def test_latency_returns_exact_percentiles(latency_client):
    """Exact nearest-rank values over real durations, so the number reported is
    one a request actually took -- not a histogram bucket estimate."""
    r = latency_client.get("/admin/usage/latency?start=2026-08-01&end=2026-08-31")
    assert r.status_code == 200
    rows = {row["day"]: row for row in r.json()["rows"]}
    day = rows["2026-08-20"]
    assert day["duration_samples"] == 4
    assert day["duration_p50"] == 200   # rank ceil(0.50*4) = 2
    assert day["duration_p95"] == 4000  # rank ceil(0.95*4) = 4
    assert day["ttft_samples"] == 4
    assert day["ttft_p95"] == 40


def test_latency_omits_a_wholly_backfilled_day(latency_client):
    """The collector reads an absent day as "no samples" and keeps whatever it
    already had, rather than writing a fabricated duration into the WBR."""
    rows = latency_client.get(
        "/admin/usage/latency?start=2026-08-01&end=2026-08-31"
    ).json()["rows"]
    assert "2026-08-19" not in {row["day"] for row in rows}


def test_latency_rejects_a_malformed_date(client):
    r = client.get("/admin/usage/latency?start=2026-08-01&end=nope")
    assert r.status_code == 400


@pytest.fixture()
def cache_client(tmp_path, monkeypatch):
    """A store written by the sampler's own writers, not by a hand-rolled INSERT.

    Every real bug in this lane came from a fixture more generous than the
    production writer, so the rows here are produced the way a sampling run
    produces them: a first sight that seeds the cursor and attributes nothing,
    a second reading that is the one that counts, and a reachable backend that
    exposes no prefix-cache series at all.
    """
    db = str(tmp_path / "usage.db")
    monkeypatch.setenv("LLM_RELAY_USAGE_DB", db)
    usage_store.reset_store_for_tests()
    conn = cache_sampler.open_cache_db(db)
    cache_sampler.record_sample(
        conn, day="2026-08-22", backend="gb200:8000", model="glm-5.2",
        current=cache_sampler.CacheCounters(1000, 300),
    )
    cache_sampler.record_sample(
        conn, day="2026-08-22", backend="gb200:8000", model="glm-5.2",
        current=cache_sampler.CacheCounters(5000, 4000),
    )
    # Production has both counterparts on the same day, and they must not
    # collapse into each other: a model whose backend reports the series but
    # whose window happened to reuse nothing (a MEASURED 0.0), and a model
    # whose backend exposes no series at all (UNKNOWN, hit_rate None).
    cache_sampler.record_sample(
        conn, day="2026-08-22", backend="mi300x:18401", model="cold-model",
        current=cache_sampler.CacheCounters(100, 0),
    )
    cache_sampler.record_sample(
        conn, day="2026-08-22", backend="mi300x:18401", model="cold-model",
        current=cache_sampler.CacheCounters(1001, 0),
    )
    cache_sampler.record_not_reported(conn, day="2026-08-22", model="unmeasured-model")
    conn.close()

    from llm_relay.api.app import create_app

    app = create_app(config_dir=_min_cfg(tmp_path))
    yield TestClient(app)
    usage_store.reset_store_for_tests()


def test_cache_returns_day_rows_and_a_per_model_summary(cache_client):
    """Token-weighted reuse, per (day, model) and rolled up per model.

    Only the delta between the two readings is attributable -- 4000 queried and
    3700 hit -- because the first observation covers an unknown span that
    predates sampling.
    """
    r = cache_client.get("/admin/usage/cache?start=2026-08-01&end=2026-08-31")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    rows = {(row["day"], row["model"]): row for row in body["rows"]}
    day = rows[("2026-08-22", "glm-5.2")]
    assert day["queried_tokens"] == 4000
    assert day["cache_read_tokens"] == 3700
    assert day["hit_rate"] == pytest.approx(0.925)
    assert day["reported"] is True
    by_model = {m["model"]: m for m in body["by_model"]}
    assert by_model["glm-5.2"]["queried_tokens"] == 4000
    assert by_model["glm-5.2"]["cache_read_tokens"] == 3700
    assert by_model["glm-5.2"]["hit_rate"] == pytest.approx(0.925)
    assert by_model["glm-5.2"]["reported"] is True


def test_cache_keeps_not_reported_distinct_from_a_measured_zero(cache_client):
    """``reported`` false means the backend exposes no prefix-cache metrics at
    all -- reuse is UNKNOWN. A hit rate of 0.0 there would read as "no reuse",
    which is the exact conflation this whole lane exists to end."""
    body = cache_client.get("/admin/usage/cache?start=2026-08-01&end=2026-08-31").json()
    rows = {(row["day"], row["model"]): row for row in body["rows"]}
    unknown = rows[("2026-08-22", "unmeasured-model")]
    assert unknown["reported"] is False
    assert unknown["queried_tokens"] == 0
    assert unknown["hit_rate"] is None
    by_model = {m["model"]: m for m in body["by_model"]}
    assert by_model["unmeasured-model"]["reported"] is False
    assert by_model["unmeasured-model"]["hit_rate"] is None
    # The counterpart: reported, queried, and genuinely no reuse. 0.0, not None.
    cold = rows[("2026-08-22", "cold-model")]
    assert cold["reported"] is True
    assert cold["queried_tokens"] == 901
    assert cold["hit_rate"] == 0.0


def test_cache_on_a_never_sampled_store_is_empty_not_an_error(client):
    """The state of every fresh deployment: the usage store exists and has
    request rows, but ``cache_daily`` is created by the sampler and the sampler
    has never run. Reading it must answer "nothing sampled", not 500 on
    ``OperationalError: no such table``."""
    r = client.get("/admin/usage/cache?start=2026-08-01&end=2026-08-31")
    assert r.status_code == 200
    body = r.json()
    assert body == {"rows": [], "by_model": [], "enabled": True, "sampled": False}


def test_cache_reports_sampled_once_the_tables_exist(cache_client):
    body = cache_client.get("/admin/usage/cache?start=2026-08-01&end=2026-08-31").json()
    assert body["sampled"] is True


def test_cache_rejects_a_malformed_date(client):
    assert client.get("/admin/usage/cache?start=2026-08-01&end=nope").status_code == 400


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
    assert c.get("/admin/usage/latency?start=2026-08-01&end=2026-08-31").json() == {
        "rows": [], "enabled": False,
    }
    assert c.get("/admin/usage/cache?start=2026-08-01&end=2026-08-31").json() == {
        "rows": [], "by_model": [], "enabled": False,
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
    for path in ("rollup", "latency", "cache"):
        q = f"/admin/usage/{path}?start=2026-08-01&end=2026-08-31"
        assert c.get(q).status_code == 401
        assert c.get(q, headers={"Authorization": "Bearer llmr_plain"}).status_code == 403
        assert c.get(q, headers={"Authorization": "Bearer llmr_admin"}).status_code == 200
