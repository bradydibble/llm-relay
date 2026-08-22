"""Salvage: recover the token history that still physically exists.

Prometheus holds per-(principal, client, model, direction) data from
2026-08-09; the KPI JSONL holds fleet-level daily totals back to 2026-06-27.
Nothing per-user exists before 2026-08-09 and historical reasoning is
unknowable (it is folded inside old completion counts), so those are recorded
as absent rather than invented.
"""
from __future__ import annotations

import pytest

from llm_relay.usage_backfill import (
    apply_request_counts,
    backfill,
    request_counts_from_prometheus,
    rows_from_kpi_record,
    rows_from_prometheus,
    synthetic_request_id,
)
from llm_relay.usage_store import open_db


def test_synthetic_ids_are_deterministic():
    a = synthetic_request_id("prom_backfill", "2026-08-20", "brady", "cc", "glm-5.2")
    b = synthetic_request_id("prom_backfill", "2026-08-20", "brady", "cc", "glm-5.2")
    c = synthetic_request_id("prom_backfill", "2026-08-21", "brady", "cc", "glm-5.2")
    assert a == b
    assert a != c


def test_prometheus_rows_map_old_direction_names():
    payload = {"data": {"result": [
        {"metric": {"principal": "brady", "client": "claude-code",
                    "model": "glm-5.2", "direction": "prompt"},
         "value": [1787000000, "1000.7"]},
        {"metric": {"principal": "brady", "client": "claude-code",
                    "model": "glm-5.2", "direction": "completion"},
         "value": [1787000000, "50.2"]},
    ]}}
    rows = rows_from_prometheus(payload, "2026-08-20")
    assert len(rows) == 1
    row = rows[0]
    assert row["input_tokens"] == 1000     # prompt -> input, truncated
    assert row["output_tokens"] == 50      # completion -> output
    assert row["reasoning_tokens"] == 0
    assert row["reasoning_source"] == "none"
    assert row["usage_source"] == "prom_backfill"
    assert row["synthetic"] == 1


def test_kpi_record_becomes_a_fleet_level_row():
    record = {"date": "2026-07-01",
              "tokens": {"prompt": 5000, "completion": 300, "reasoning": 0}}
    rows = rows_from_kpi_record(record)
    assert len(rows) == 1
    assert rows[0]["principal"] == ""      # fleet-level: no per-user attribution
    assert rows[0]["input_tokens"] == 5000
    assert rows[0]["output_tokens"] == 300
    assert rows[0]["usage_source"] == "kpi_backfill"


def test_kpi_record_reads_the_legacy_by_type_shape():
    record = {"date": "2026-06-27",
              "tokens": {"by_type": {"prompt": 10, "completion": 2}}}
    rows = rows_from_kpi_record(record)
    assert rows[0]["input_tokens"] == 10
    assert rows[0]["output_tokens"] == 2


def test_kpi_record_with_no_tokens_is_skipped():
    assert rows_from_kpi_record({"date": "2026-07-02", "tokens": {}}) == []


def test_backfill_is_idempotent(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    rows = rows_from_kpi_record(
        {"date": "2026-07-01", "tokens": {"prompt": 5000, "completion": 300}}
    )
    assert backfill(conn, rows) == 1
    assert backfill(conn, rows) == 0  # re-run inserts nothing
    total = conn.execute("SELECT SUM(input_tokens) FROM requests").fetchone()[0]
    assert total == 5000


def test_backfill_never_overwrites_a_live_row(tmp_path):
    conn = open_db(str(tmp_path / "u.db"))
    conn.execute(
        "INSERT INTO requests (request_id, ts, day, principal, client, model, "
        "provider, outcome, streamed, input_tokens, output_tokens, "
        "reasoning_tokens, usage_source, reasoning_source) VALUES "
        "('live-1', 1.0, '2026-08-20', 'brady', 'cc', 'glm-5.2', 'gb200', "
        "'success', 1, 99, 9, 0, 'upstream_final', 'none')"
    )
    rows = rows_from_prometheus(
        {"data": {"result": [
            {"metric": {"principal": "brady", "client": "cc", "model": "glm-5.2",
                        "direction": "prompt"}, "value": [1, "500"]}]}},
        "2026-08-20",
    )
    backfill(conn, rows)
    live = conn.execute(
        "SELECT input_tokens FROM requests WHERE request_id = 'live-1'"
    ).fetchone()[0]
    assert live == 99  # untouched


def test_illegal_row_is_rejected_loudly_not_swallowed(tmp_path):
    # INSERT OR IGNORE ignores every constraint failure, not just PK conflicts,
    # so a CHECK violation would otherwise be indistinguishable from an
    # idempotent skip. A backfill that silently drops history is the bug.
    conn = open_db(str(tmp_path / "u.db"))
    bad = rows_from_kpi_record(
        {"date": "2026-07-01", "tokens": {"prompt": 10, "completion": 5}}
    )[0]
    bad["reasoning_tokens"] = 999  # exceeds output_tokens
    with pytest.raises(ValueError):
        backfill(conn, [bad])


# --- request counts on synthetic daily rows ----------------------------------
# A backfilled row is one daily aggregate per (day, principal, client, model),
# so COUNT(*) reports 1 where the real day had thousands of calls. The real
# count comes from Prometheus at backfill time and rides in the row.


def test_request_counts_from_prometheus_parses_per_model_totals():
    payload = {"data": {"result": [
        {"metric": {"model": "glm-5.2"}, "value": [1787000000, "3181.4"]},
        {"metric": {"model": "ornith-35b"}, "value": [1787000000, "12.6"]},
        {"metric": {"model": "none"}, "value": [1787000000, "5"]},
        {"metric": {}, "value": [1787000000, "9"]},
    ]}}
    counts = request_counts_from_prometheus(payload)
    assert counts == {"glm-5.2": 3181, "ornith-35b": 13}  # rounded, not truncated


def test_daily_count_splits_across_rows_in_proportion_to_tokens():
    rows = rows_from_prometheus({"data": {"result": [
        {"metric": {"principal": "brady", "client": "cc", "model": "glm-5.2",
                    "direction": "prompt"}, "value": [1, "7500"]},
        {"metric": {"principal": "jrodriguez", "client": "vscode",
                    "model": "glm-5.2", "direction": "prompt"},
         "value": [1, "2500"]},
    ]}}, "2026-08-20")
    applied = apply_request_counts(rows, {"glm-5.2": 101})
    assert applied == 1  # one model's count was applied
    by_principal = {r["principal"]: r["request_count"] for r in rows}
    # 75% / 25% of 101 -> 75.75 / 25.25; floors are 75 and 25, and the leftover
    # unit lands on the largest row so the day's total stays exactly 101.
    assert by_principal == {"brady": 76, "jrodriguez": 25}
    assert sum(by_principal.values()) == 101


def test_a_model_prometheus_did_not_report_keeps_a_count_of_one():
    """Inventing a number for a model the counter never saw would be worse than
    an obvious 1: the 1 is visibly a floor, a guess is not."""
    rows = rows_from_prometheus({"data": {"result": [
        {"metric": {"principal": "brady", "client": "cc", "model": "glm-5.2",
                    "direction": "prompt"}, "value": [1, "100"]},
    ]}}, "2026-08-20")
    assert apply_request_counts(rows, {"some-other-model": 50}) == 0
    assert rows[0]["request_count"] == 1


def test_backfill_persists_the_request_count_it_was_given(tmp_path):
    """Built from a row shaped exactly as ``rows_from_prometheus`` emits it: if
    request_count were missing from the insert column list the value would
    silently revert to the column default and the fix would do nothing."""
    conn = open_db(str(tmp_path / "u.db"))
    rows = rows_from_prometheus({"data": {"result": [
        {"metric": {"principal": "brady", "client": "cc", "model": "glm-5.2",
                    "direction": "prompt"}, "value": [1, "1000"]},
        {"metric": {"principal": "brady", "client": "cc", "model": "glm-5.2",
                    "direction": "completion"}, "value": [1, "100"]},
    ]}}, "2026-08-20")
    apply_request_counts(rows, {"glm-5.2": 3181})
    assert backfill(conn, rows) == 1
    got = conn.execute(
        "SELECT request_count, synthetic FROM requests"
    ).fetchone()
    assert got == (3181, 1)


def test_synthetic_rows_start_at_one_request(tmp_path):
    rows = rows_from_kpi_record(
        {"date": "2026-07-01", "tokens": {"prompt": 5000, "completion": 300}}
    )
    assert rows[0]["request_count"] == 1


def test_kpi_record_carries_its_own_fleet_request_total():
    """The KPI JSONL already records the day's request total, so a fleet row has
    no reason to sit at the floor of 1."""
    rows = rows_from_kpi_record({
        "date": "2026-07-01",
        "tokens": {"prompt": 5000, "completion": 300},
        "requests": {"total": 4212, "terminal": 6},
    })
    assert rows[0]["request_count"] == 4212
