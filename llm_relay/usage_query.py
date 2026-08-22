"""Read-side aggregation over the usage store.

Pure against an injected connection so the SQL is testable without a running
relay. Every consumer downstream (admin cost tab, per-user usage, WBR
collector, users overview) is served by ``rollup`` and ``summary``, so there is
one aggregation path rather than four divergent ones.
"""
from __future__ import annotations

import os
import re
import sqlite3

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Sources whose numbers came from the upstream and are exact.
_EXACT = ("upstream_incremental", "upstream_final")

# No tokens were consumed, so the request is neither measured nor estimated.
_NO_USAGE = "none"


def valid_day(value: str) -> bool:
    return bool(_DAY_RE.match(value or ""))


def rollup(conn: sqlite3.Connection, start_day: str, end_day: str) -> list[dict]:
    """Per (day, principal, client, model, alias) token totals in a date window.

    ``exact_requests`` and ``estimated_requests`` split the same window by how
    its numbers were learned, so data quality travels with the data and a UI can
    state the measured share instead of implying every figure is exact. They do
    not have to sum to ``requests``: a ``usage_source`` of ``none`` means no
    tokens were consumed and counts as neither.
    """
    placeholders = ", ".join("?" for _ in _EXACT)
    cur = conn.execute(
        "SELECT day, principal, client, model, alias, COUNT(*) AS requests, "
        "SUM(input_tokens), SUM(output_tokens), SUM(reasoning_tokens), "
        "SUM(cache_read_tokens), "
        f"SUM(CASE WHEN usage_source IN ({placeholders}) THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN usage_source NOT IN ({placeholders}) "
        "         AND usage_source != ? THEN 1 ELSE 0 END) "
        "FROM requests WHERE day >= ? AND day <= ? "
        "GROUP BY day, principal, client, model, alias "
        "ORDER BY day, principal, model",
        (*_EXACT, *_EXACT, _NO_USAGE, start_day, end_day),
    )
    return [
        {
            "day": r[0], "principal": r[1], "client": r[2], "model": r[3],
            "alias": r[4], "requests": int(r[5] or 0),
            "input_tokens": int(r[6] or 0), "output_tokens": int(r[7] or 0),
            "reasoning_tokens": int(r[8] or 0), "cache_read_tokens": int(r[9] or 0),
            "exact_requests": int(r[10] or 0), "estimated_requests": int(r[11] or 0),
        }
        for r in cur.fetchall()
    ]


def summary(conn: sqlite3.Connection) -> dict:
    """All-time totals and true first/last activity per principal.

    ``last_activity_ts`` is the maximum event timestamp — an actual request
    time, unlike the Prometheus ``timestamp()`` this replaces, which reported
    scrape time and so read "seconds ago" for anyone with a live series.
    """
    cur = conn.execute(
        "SELECT principal, COUNT(*), SUM(input_tokens), SUM(output_tokens), "
        "SUM(reasoning_tokens), MIN(ts), MAX(ts) "
        "FROM requests GROUP BY principal ORDER BY principal"
    )
    by_principal = {}
    for r in cur.fetchall():
        by_principal[r[0]] = {
            "requests": int(r[1] or 0),
            "all_time_input_tokens": int(r[2] or 0),
            "all_time_output_tokens": int(r[3] or 0),
            "all_time_reasoning_tokens": int(r[4] or 0),
            "first_seen_ts": r[5],
            "last_activity_ts": r[6],
        }
    return {"by_principal": by_principal}


def store_health(conn: sqlite3.Connection, path: str) -> dict:
    """Row count, distinct days, and on-disk size — growth must be observable."""
    rows = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    days = conn.execute("SELECT COUNT(DISTINCT day) FROM requests").fetchone()[0]
    size = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            size += os.path.getsize(path + suffix)
        except OSError:
            pass
    return {"rows": int(rows or 0), "days": int(days or 0), "bytes": size}
