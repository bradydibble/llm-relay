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

# The one outcome that is not a failure. Everything else -- upstream errors,
# client aborts, saturation, context rejects -- is counted as one.
_SUCCESS = "success"


def valid_day(value: str) -> bool:
    return bool(_DAY_RE.match(value or ""))


def rollup(conn: sqlite3.Connection, start_day: str, end_day: str) -> list[dict]:
    """Per (day, principal, client, model, alias) totals in a date window.

    Every count is a sum of ``request_count``, never ``COUNT(*)``. A live row
    accounts for one request; a synthetic backfill row is a whole day's
    aggregate, so counting rows reported 1 where the day really had thousands --
    which is why request totals used to have to come back from Prometheus.

    ``exact_requests`` and ``estimated_requests`` split the same window by how
    its numbers were learned, so data quality travels with the data and a UI can
    state the measured share instead of implying every figure is exact. They do
    not have to sum to ``requests``: a ``usage_source`` of ``none`` means no
    tokens were consumed and counts as neither.

    ``successes``, ``failures`` and the ``outcomes`` map answer what used to
    take a separate Prometheus query. Caveat worth surfacing in a UI: synthetic
    rows all carry ``outcome='success'`` because per-outcome history was not
    recoverable, so a backfilled day reports no failures rather than none having
    happened.
    """
    placeholders = ", ".join("?" for _ in _EXACT)
    cur = conn.execute(
        "SELECT day, principal, client, model, alias, outcome, "
        "SUM(request_count) AS requests, "
        "SUM(input_tokens), SUM(output_tokens), SUM(reasoning_tokens), "
        "SUM(cache_read_tokens), "
        f"SUM(CASE WHEN usage_source IN ({placeholders}) "
        "         THEN request_count ELSE 0 END), "
        f"SUM(CASE WHEN usage_source NOT IN ({placeholders}) "
        "         AND usage_source != ? THEN request_count ELSE 0 END) "
        "FROM requests WHERE day >= ? AND day <= ? "
        "GROUP BY day, principal, client, model, alias, outcome "
        "ORDER BY day, principal, model, client, alias",
        (*_EXACT, *_EXACT, _NO_USAGE, start_day, end_day),
    )
    # Grouped one level finer than the result grain (by outcome) and folded here:
    # one table scan serves both the totals and the outcome breakdown, so the
    # two can never disagree the way two separate queries could.
    grouped: dict[tuple, dict] = {}
    for r in cur.fetchall():
        key = (r[0], r[1], r[2], r[3], r[4])
        row = grouped.get(key)
        if row is None:
            row = {
                "day": r[0], "principal": r[1], "client": r[2], "model": r[3],
                "alias": r[4], "requests": 0,
                "input_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "cache_read_tokens": 0,
                "exact_requests": 0, "estimated_requests": 0,
                "successes": 0, "failures": 0, "outcomes": {},
            }
            grouped[key] = row
        outcome = r[5] or ""
        count = int(r[6] or 0)
        row["requests"] += count
        row["outcomes"][outcome] = row["outcomes"].get(outcome, 0) + count
        if outcome == _SUCCESS:
            row["successes"] += count
        else:
            row["failures"] += count
        row["input_tokens"] += int(r[7] or 0)
        row["output_tokens"] += int(r[8] or 0)
        row["reasoning_tokens"] += int(r[9] or 0)
        row["cache_read_tokens"] += int(r[10] or 0)
        row["exact_requests"] += int(r[11] or 0)
        row["estimated_requests"] += int(r[12] or 0)
    return list(grouped.values())


# Nearest-rank percentile over the day's sorted observations: the value at
# 1-based rank ceil(p/100 * n). Integer-only, so ``(n*p + 99) / 100`` under
# SQLite's truncating integer division *is* that ceiling. No interpolation and
# no bucket boundaries -- the result is always a duration something actually
# took, which is the whole advantage over a Prometheus histogram_quantile
# estimate (bucketed, and only as precise as the bucket layout).
_PERCENTILE_SQL = """
WITH observed AS (
    SELECT day, {column} AS v FROM requests
    WHERE day >= ? AND day <= ? AND synthetic = 0 AND {column} IS NOT NULL
), ranked AS (
    SELECT day, v,
           ROW_NUMBER() OVER (PARTITION BY day ORDER BY v) AS rn,
           COUNT(*)     OVER (PARTITION BY day)           AS n
    FROM observed
)
SELECT day, n,
       MAX(CASE WHEN rn = (n * 50 + 99) / 100 THEN v END) AS p50,
       MAX(CASE WHEN rn = (n * 95 + 99) / 100 THEN v END) AS p95
FROM ranked GROUP BY day ORDER BY day
"""

# Only these may reach the format string above.
_LATENCY_COLUMNS = ("duration_ms", "ttft_ms")


def _percentiles(conn: sqlite3.Connection, column: str,
                 start_day: str, end_day: str) -> dict[str, tuple]:
    if column not in _LATENCY_COLUMNS:
        raise ValueError(f"not a latency column: {column}")
    cur = conn.execute(
        _PERCENTILE_SQL.format(column=column), (start_day, end_day)
    )
    return {r[0]: (int(r[1] or 0), r[2], r[3]) for r in cur.fetchall()}


def latency(conn: sqlite3.Connection, start_day: str,
            end_day: str) -> list[dict]:
    """Per-day exact latency percentiles from the observations themselves.

    Every request already stores its own ``duration_ms`` and ``ttft_ms``, so
    these are exact percentiles over real values -- strictly better than the
    bucketed histogram estimate they replace. NULLs are excluded (a
    non-streaming or failed-early request has no time to first token) and
    synthetic backfill rows are excluded entirely: they were never observed, and
    one fabricated duration would move a percentile nobody could then explain.

    ``duration_samples`` / ``ttft_samples`` travel with the percentiles so a
    reader can see how many observations are behind them -- a p95 over four
    requests deserves less trust than one over four thousand. A day with no
    observations of either kind is absent rather than reported as zero.
    """
    duration = _percentiles(conn, "duration_ms", start_day, end_day)
    ttft = _percentiles(conn, "ttft_ms", start_day, end_day)
    rows = []
    for day in sorted(set(duration) | set(ttft)):
        d = duration.get(day) or (0, None, None)
        t = ttft.get(day) or (0, None, None)
        rows.append({
            "day": day,
            "duration_samples": d[0], "duration_p50": d[1], "duration_p95": d[2],
            "ttft_samples": t[0], "ttft_p50": t[1], "ttft_p95": t[2],
        })
    return rows


def summary(conn: sqlite3.Connection) -> dict:
    """All-time totals and true first/last activity per principal.

    ``last_activity_ts`` is the maximum event timestamp — an actual request
    time, unlike the Prometheus ``timestamp()`` this replaces, which reported
    scrape time and so read "seconds ago" for anyone with a live series.

    ``requests`` sums ``request_count`` for the same reason :func:`rollup` does:
    a backfilled row stands for a whole day, so counting rows would report a
    heavy user's history as a handful of calls.
    """
    cur = conn.execute(
        "SELECT principal, SUM(request_count), SUM(input_tokens), "
        "SUM(output_tokens), SUM(reasoning_tokens), MIN(ts), MAX(ts) "
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
