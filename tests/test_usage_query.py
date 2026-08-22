"""Read-side aggregation over the usage store."""
from __future__ import annotations

from llm_relay.usage_query import latency, rollup, summary
from llm_relay.usage_store import open_db


def _seed(conn, rows):
    for r in rows:
        conn.execute(
            "INSERT INTO requests (request_id, ts, day, principal, client, alias, "
            "model, provider, outcome, streamed, input_tokens, output_tokens, "
            "reasoning_tokens, cache_read_tokens, usage_source, reasoning_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r,
        )


def _live(conn, request_id, **over):
    """A row shaped like the live writer's: synthetic=0, request_count absent so
    the column default applies, exactly as instrumentation.py inserts it."""
    row = {
        "request_id": request_id, "ts": 1.0, "day": "2026-08-20",
        "principal": "brady", "client": "claude-code", "alias": "main",
        "model": "glm-5.2", "provider": "gb200", "outcome": "success",
        "streamed": 1, "duration_ms": None, "ttft_ms": None,
        "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0,
        "cache_read_tokens": 0, "usage_source": "upstream_final",
        "reasoning_source": "none", "synthetic": 0,
    }
    row.update(over)
    cols = ", ".join(row)
    conn.execute(
        f"INSERT INTO requests ({cols}) VALUES "
        f"({', '.join('?' for _ in row)})", tuple(row.values()),
    )


def _synthetic(conn, request_id, **over):
    """A row shaped like usage_backfill's: one daily aggregate, synthetic=1,
    no latency observations, and a real request_count above 1."""
    row = {
        "request_id": request_id, "ts": 1.0, "day": "2026-08-20",
        "principal": "brady", "client": "claude-code", "alias": None,
        "model": "glm-5.2", "provider": "", "outcome": "success",
        "streamed": 0, "duration_ms": None, "ttft_ms": None,
        "input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": 0,
        "cache_read_tokens": 0, "usage_source": "prom_backfill",
        "reasoning_source": "none", "synthetic": 1, "request_count": 3181,
    }
    row.update(over)
    cols = ", ".join(row)
    conn.execute(
        f"INSERT INTO requests ({cols}) VALUES "
        f"({', '.join('?' for _ in row)})", tuple(row.values()),
    )


def test_rollup_groups_and_sums(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _seed(conn, [
        ("r1", 1.0, "2026-08-20", "brady", "claude-code", "main", "glm-5.2",
         "gb200", "success", 1, 1000, 100, 40, 0, "upstream_final", "char_split"),
        ("r2", 2.0, "2026-08-20", "brady", "claude-code", "main", "glm-5.2",
         "gb200", "success", 1, 500, 50, 10, 0, "upstream_final", "char_split"),
        ("r3", 3.0, "2026-08-21", "jrodriguez", "vscode", "fast", "ornith-35b",
         "llama-01", "success", 1, 7, 3, 0, 0, "frame_count", "none"),
    ])
    rows = rollup(conn, "2026-08-01", "2026-08-31")
    assert len(rows) == 2
    first = [r for r in rows if r["principal"] == "brady"][0]
    assert first["input_tokens"] == 1500
    assert first["output_tokens"] == 150
    assert first["reasoning_tokens"] == 50
    assert first["requests"] == 2


def test_rollup_separates_exact_from_estimated(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _seed(conn, [
        ("a", 1.0, "2026-08-20", "p", "c", None, "m", "pr", "success", 1,
         10, 1, 0, 0, "upstream_incremental", "none"),
        ("b", 2.0, "2026-08-20", "p", "c", None, "m", "pr", "success", 1,
         10, 1, 0, 0, "tokenizer_estimate", "none"),
    ])
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["exact_requests"] == 1
    assert row["estimated_requests"] == 1


def test_rollup_counts_a_zero_token_request_as_neither(tmp_path):
    """``usage_source='none'`` means no tokens were consumed, so the request is
    neither measured nor estimated: counting it either way would misreport the
    measured share of a window."""
    conn = open_db(str(tmp_path / "u.db"))
    _seed(conn, [
        ("a", 1.0, "2026-08-20", "p", "c", None, "m", "pr", "error", 1,
         0, 0, 0, 0, "none", "none"),
    ])
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 1
    assert row["exact_requests"] == 0
    assert row["estimated_requests"] == 0


def test_rollup_respects_the_date_window(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _seed(conn, [
        ("a", 1.0, "2026-07-01", "p", "c", None, "m", "pr", "success", 1,
         10, 1, 0, 0, "upstream_final", "none"),
    ])
    assert rollup(conn, "2026-08-01", "2026-08-31") == []


def test_summary_reports_all_time_and_true_last_activity(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _seed(conn, [
        ("a", 100.0, "2026-08-20", "brady", "c", None, "m", "pr", "success", 1,
         1000, 10, 0, 0, "upstream_final", "none"),
        ("b", 900.0, "2026-08-21", "brady", "c", None, "m", "pr", "success", 1,
         2000, 20, 5, 0, "upstream_final", "char_split"),
    ])
    s = summary(conn)
    me = s["by_principal"]["brady"]
    assert me["all_time_input_tokens"] == 3000
    assert me["all_time_output_tokens"] == 30
    assert me["all_time_reasoning_tokens"] == 5
    # Real event time, not a scrape timestamp.
    assert me["last_activity_ts"] == 900.0
    assert me["first_seen_ts"] == 100.0
    assert me["requests"] == 2


def test_summary_is_empty_not_broken_on_a_fresh_database(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    s = summary(conn)
    assert s["by_principal"] == {}


# --- requests come from request_count, not COUNT(*) ---------------------------


def test_rollup_sums_request_count_on_a_backfilled_day(tmp_path):
    """A synthetic row is one daily aggregate covering thousands of calls.
    COUNT(*) reported 1, which is why request totals had to run back through
    Prometheus; summing request_count is what removes that need."""
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181)
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 3181


def test_rollup_still_counts_one_per_live_row(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "l1")
    _live(conn, "l2")
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 2


def test_rollup_mixes_live_and_backfilled_rows_without_double_counting(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181)
    _live(conn, "l1", alias=None, usage_source="prom_backfill")
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 3182  # 3181 aggregated + 1 real


# --- outcomes in the rollup --------------------------------------------------


def test_rollup_reports_successes_failures_and_the_outcome_map(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "ok1")
    _live(conn, "ok2")
    _live(conn, "bad1", outcome="upstream_error")
    _live(conn, "bad2", outcome="upstream_error")
    _live(conn, "bad3", outcome="client_abort")
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 5
    assert row["successes"] == 2
    assert row["failures"] == 3
    assert row["outcomes"] == {
        "success": 2, "upstream_error": 2, "client_abort": 1,
    }


def test_rollup_outcomes_weight_by_request_count(tmp_path):
    """A synthetic row carries outcome='success' for its whole aggregate -- a
    known limitation, not a claim that 3181 calls all succeeded."""
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181)
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["successes"] == 3181
    assert row["failures"] == 0
    assert row["outcomes"] == {"success": 3181}
    # The quality split is in the same unit as the counts, so a backfilled day
    # reads as wholly estimated rather than as one estimated call in 3,181.
    assert row["estimated_requests"] == 3181
    assert row["exact_requests"] == 0


def test_rollup_counts_fallbacks_from_the_fell_back_column(tmp_path):
    """The last usage number Prometheus still answered. The column was already
    written per request; nothing read it, so the collector had to keep a
    ``llm_relay_fallbacks_total`` query alive purely for this one field."""
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "a", fell_back=1)
    _live(conn, "b", fell_back=0)
    _live(conn, "c", fell_back=None)  # pre-column history stays uncounted
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 3
    assert row["fallbacks"] == 1


def test_rollup_fallbacks_weight_by_request_count(tmp_path):
    """Same unit as every other count in the row: an aggregate row stands for
    its whole day, so COUNT(*) would report one fallback for thousands."""
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181, fell_back=1)
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["fallbacks"] == 3181


def test_rollup_reports_no_fallbacks_on_a_backfilled_day(tmp_path):
    """Backfill could not recover which requests fell back, so it wrote NULL.
    A backfilled day therefore reports zero fallbacks rather than none having
    happened -- the same caveat the outcome map carries, and the reason a
    consumer must decide per day whether to trust it."""
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181)
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 3181
    assert row["fallbacks"] == 0


def test_summary_sums_request_count_too(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _synthetic(conn, "s1", request_count=3181)
    _live(conn, "l1")
    assert summary(conn)["by_principal"]["brady"]["requests"] == 3182


def test_rollup_keeps_the_exact_estimated_split_across_outcomes(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "a", usage_source="upstream_incremental")
    _live(conn, "b", usage_source="tokenizer_estimate", outcome="upstream_error")
    _live(conn, "c", usage_source="none", outcome="client_abort",
          input_tokens=0, output_tokens=0)
    row = rollup(conn, "2026-08-20", "2026-08-20")[0]
    assert row["requests"] == 3
    assert row["exact_requests"] == 1
    assert row["estimated_requests"] == 1
    assert row["failures"] == 2


# --- exact latency percentiles from real observations ------------------------


def test_latency_percentiles_are_exact_nearest_rank(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    for i, ms in enumerate((30, 10, 40, 20)):  # inserted out of order on purpose
        _live(conn, f"d{i}", duration_ms=ms)
    rows = latency(conn, "2026-08-20", "2026-08-20")
    assert len(rows) == 1
    r = rows[0]
    assert r["day"] == "2026-08-20"
    assert r["duration_samples"] == 4
    # Nearest-rank: ceil(p/100 * n) as a 1-based rank over the sorted values.
    assert r["duration_p50"] == 20    # rank ceil(0.50*4) = 2
    assert r["duration_p95"] == 40    # rank ceil(0.95*4) = 4


def test_latency_ignores_nulls_and_reports_the_sample_count(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "a", duration_ms=100, ttft_ms=5)
    _live(conn, "b", duration_ms=200, ttft_ms=15)
    _live(conn, "c", duration_ms=300, ttft_ms=None)  # no first token measured
    r = latency(conn, "2026-08-20", "2026-08-20")[0]
    assert r["duration_samples"] == 3
    assert r["ttft_samples"] == 2
    assert r["ttft_p50"] == 5     # rank ceil(0.50*2) = 1
    assert r["ttft_p95"] == 15    # rank ceil(0.95*2) = 2


def test_latency_excludes_synthetic_rows(tmp_path):
    """Backfilled rows have no latency at all; a fabricated duration on one
    would poison the percentile with a number nobody observed."""
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "real", duration_ms=100)
    _synthetic(conn, "fake", duration_ms=999999)
    r = latency(conn, "2026-08-20", "2026-08-20")[0]
    assert r["duration_samples"] == 1
    assert r["duration_p50"] == 100
    assert r["duration_p95"] == 100


def test_latency_reports_a_day_with_no_observations_at_all(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "a", duration_ms=None, ttft_ms=None)
    assert latency(conn, "2026-08-20", "2026-08-20") == []


def test_latency_separates_days_and_respects_the_window(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    _live(conn, "a", day="2026-08-20", duration_ms=100)
    _live(conn, "b", day="2026-08-21", duration_ms=900)
    _live(conn, "c", day="2026-08-22", duration_ms=7)
    rows = latency(conn, "2026-08-20", "2026-08-21")
    assert [r["day"] for r in rows] == ["2026-08-20", "2026-08-21"]
    assert [r["duration_p50"] for r in rows] == [100, 900]


def test_latency_is_empty_not_broken_on_a_fresh_database(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    assert latency(conn, "2026-08-01", "2026-08-31") == []
