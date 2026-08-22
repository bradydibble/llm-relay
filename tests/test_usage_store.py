"""The durable usage store: exact rows that survive a relay restart.

Prometheus counters reset on restart and increase() extrapolates, which is why
the admin tab showed 40M all-time against 300M in one day. This store is the
system of record; it must be exact, single-writer, and never able to break a
request.
"""
from __future__ import annotations

import sqlite3

import pytest

from llm_relay.usage_store import UsageStore, open_db


def _row(**over):
    row = {
        "request_id": "req-1",
        "ts": 1787000000.0,
        "day": "2026-08-20",
        "principal": "brady",
        "client": "claude-code",
        "alias": "main",
        "model": "glm-5.2",
        "provider": "gb200",
        "outcome": "success",
        "streamed": 1,
        "duration_ms": 1500,
        "ttft_ms": 300,
        "input_tokens": 120000,
        "output_tokens": 900,
        "reasoning_tokens": 400,
        "cache_read_tokens": 0,
        "usage_source": "upstream_incremental",
        "reasoning_source": "upstream_details",
        "synthetic": 0,
        "message_count": 12,
        "system_hash": "abc",
        "prefix_hash": "def",
        "tool_count": 3,
        "temperature": 0.2,
        "max_tokens": 4096,
        "confidentiality": "non_confidential",
        "fell_back": 0,
    }
    row.update(over)
    return row


def test_record_persists_a_row(tmp_path):
    store = UsageStore(str(tmp_path / "usage.db"))
    try:
        store.record(_row())
        store.flush()
        conn = open_db(str(tmp_path / "usage.db"))
        got = conn.execute(
            "SELECT principal, input_tokens, output_tokens, reasoning_tokens "
            "FROM requests WHERE request_id = 'req-1'"
        ).fetchone()
        assert got == ("brady", 120000, 900, 400)
    finally:
        store.close()


def test_rows_survive_reopening_the_database(tmp_path):
    path = str(tmp_path / "usage.db")
    store = UsageStore(path)
    store.record(_row())
    store.flush()
    store.close()

    store2 = UsageStore(path)
    try:
        conn = open_db(path)
        total = conn.execute("SELECT SUM(input_tokens) FROM requests").fetchone()[0]
        assert total == 120000
    finally:
        store2.close()


def test_duplicate_request_id_is_ignored_not_doubled(tmp_path):
    store = UsageStore(str(tmp_path / "usage.db"))
    try:
        store.record(_row())
        store.record(_row())  # same deterministic id, e.g. a re-run backfill
        store.flush()
        conn = open_db(str(tmp_path / "usage.db"))
        n, total = conn.execute(
            "SELECT COUNT(*), SUM(input_tokens) FROM requests"
        ).fetchone()
        assert (n, total) == (1, 120000)
    finally:
        store.close()


def test_reasoning_never_exceeds_output_is_enforced_by_schema(tmp_path):
    conn = open_db(str(tmp_path / "usage.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO requests (request_id, ts, day, principal, client, model, "
            "provider, outcome, streamed, input_tokens, output_tokens, "
            "reasoning_tokens, usage_source, reasoning_source) VALUES "
            "('bad', 1.0, '2026-08-20', 'p', 'c', 'm', 'pr', 'success', 1, "
            "10, 100, 500, 'upstream_final', 'upstream_details')"
        )


def test_reasoning_over_output_is_counted_not_silently_swallowed(tmp_path):
    # INSERT OR IGNORE (needed so a re-run backfill dedupes on request_id) makes
    # SQLite ignore *every* constraint failure, so without an explicit check a
    # row violating the of-which invariant would disappear with no error and no
    # count. Silent loss is the failure mode this store exists to end.
    store = UsageStore(str(tmp_path / "usage.db"))
    try:
        store.record(_row(output_tokens=10, reasoning_tokens=400))
        store.flush()
        assert store.dropped >= 1
        conn = open_db(str(tmp_path / "usage.db"))
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    finally:
        store.close()


def test_a_bad_row_does_not_raise_into_the_caller(tmp_path):
    # Telemetry must never break a request: a malformed row is dropped, counted.
    store = UsageStore(str(tmp_path / "usage.db"))
    try:
        store.record({"request_id": "only-a-key"})  # missing required columns
        store.flush()
        assert store.dropped >= 1
    finally:
        store.close()


def test_record_after_close_is_silent(tmp_path):
    store = UsageStore(str(tmp_path / "usage.db"))
    store.close()
    store.record(_row())  # must not raise
    assert store.dropped >= 1


def test_unwritable_path_degrades_silently(tmp_path):
    store = UsageStore(str(tmp_path / "no-such-dir" / "nested" / "usage.db"))
    try:
        store.record(_row())
        store.flush()
    finally:
        store.close()  # must not raise


def test_queue_saturation_counts_drops_instead_of_blocking(tmp_path):
    store = UsageStore(str(tmp_path / "usage.db"), maxsize=2, autostart=False)
    try:
        for i in range(50):
            store.record(_row(request_id=f"req-{i}"))
        assert store.dropped > 0  # bounded queue shed load rather than blocking
    finally:
        store.close()


def test_rollup_aggregates_by_day_principal_model(tmp_path):
    store = UsageStore(str(tmp_path / "usage.db"))
    try:
        # reasoning_tokens is an of-which subset of output_tokens and the schema
        # enforces it, so these small-output rows must carry a legal reasoning
        # value or they would be rejected rather than aggregated.
        store.record(_row(request_id="a", input_tokens=100, output_tokens=10,
                          reasoning_tokens=0))
        store.record(_row(request_id="b", input_tokens=200, output_tokens=20,
                          reasoning_tokens=0))
        store.record(_row(request_id="c", day="2026-08-21", input_tokens=5,
                          output_tokens=1, reasoning_tokens=0))
        store.flush()
        conn = open_db(str(tmp_path / "usage.db"))
        rows = conn.execute(
            "SELECT day, SUM(input_tokens), SUM(output_tokens), COUNT(*) "
            "FROM requests GROUP BY day ORDER BY day"
        ).fetchall()
        assert rows == [("2026-08-20", 300, 30, 2), ("2026-08-21", 5, 1, 1)]
    finally:
        store.close()
