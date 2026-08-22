"""Read-side aggregation over the usage store."""
from __future__ import annotations

from llm_relay.usage_query import rollup, summary
from llm_relay.usage_store import open_db


def _seed(conn, rows):
    for r in rows:
        conn.execute(
            "INSERT INTO requests (request_id, ts, day, principal, client, alias, "
            "model, provider, outcome, streamed, input_tokens, output_tokens, "
            "reasoning_tokens, cache_read_tokens, usage_source, reasoning_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r,
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
