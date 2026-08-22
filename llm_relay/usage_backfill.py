"""Recover token history from the two places it still exists.

Prometheus retains per-(principal, client, model, direction) series from
2026-08-09; the portal's KPI JSONL holds fleet-level daily totals back to
2026-06-27. Rows written here are ``synthetic=1`` daily aggregates, not real
per-request rows, so per-request analytics can exclude them.

Historical reasoning tokens are NOT recoverable: they are folded inside the old
``completion`` counts and no query can separate them. They are recorded as 0
with ``reasoning_source='none'`` so the UI can say "not measured" rather than
implying a real zero.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import urllib.parse
import urllib.request

SOURCE_PROM = "prom_backfill"
SOURCE_KPI = "kpi_backfill"

# The daily per-user series Prometheus still holds. ``increase()`` extrapolates,
# which is exactly why the live path writes exact integers to SQLite instead --
# but for history already lost it is the only number that exists.
PROM_QUERY = (
    "sum by (principal, client, model, direction) "
    "(increase(llm_relay_tokens_total[1d]))"
)

# Old direction label -> standard name.
_DIRECTION_MAP = {
    "prompt": "input_tokens", "input": "input_tokens",
    "completion": "output_tokens", "output": "output_tokens",
}


def synthetic_request_id(source: str, day: str, principal: str,
                         client: str, model: str) -> str:
    """Deterministic id so a re-run is a no-op instead of a doubling."""
    key = "|".join((source, day, principal or "", client or "", model or ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _base_row(source: str, day: str, principal: str, client: str,
              model: str) -> dict:
    ts = dt.datetime.strptime(day, "%Y-%m-%d").replace(
        tzinfo=dt.timezone.utc, hour=12
    ).timestamp()
    return {
        "request_id": synthetic_request_id(source, day, principal, client, model),
        "ts": ts, "day": day, "principal": principal, "client": client,
        "alias": None, "model": model, "provider": "", "outcome": "success",
        "streamed": 0, "duration_ms": None, "ttft_ms": None,
        "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
        "cache_read_tokens": 0, "usage_source": source,
        "reasoning_source": "none", "synthetic": 1, "message_count": None,
        "system_hash": None, "prefix_hash": None, "tool_count": None,
        "temperature": None, "max_tokens": None, "confidentiality": None,
        "fell_back": None,
    }


def rows_from_prometheus(payload: dict, day: str) -> list[dict]:
    """Convert one instant-query payload grouped by
    (principal, client, model, direction) into daily rows."""
    merged: dict[tuple, dict] = {}
    for series in ((payload or {}).get("data") or {}).get("result") or []:
        m = series.get("metric") or {}
        direction = str(m.get("direction") or "prompt")
        field = _DIRECTION_MAP.get(direction)
        if field is None:
            continue  # a 'reasoning' series cannot exist pre-cutover
        model = str(m.get("model") or "")
        if not model or model == "none":
            continue
        principal = str(m.get("principal") or "")
        client = str(m.get("client") or "")
        try:
            tokens = int(float(series.get("value", [None, "0"])[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if tokens <= 0:
            continue
        key = (day, principal, client, model)
        row = merged.get(key) or _base_row(SOURCE_PROM, day, principal, client, model)
        row[field] += tokens
        merged[key] = row
    return list(merged.values())


def rows_from_kpi_record(record: dict) -> list[dict]:
    """Convert one KPI JSONL daily record into a single fleet-level row.

    Handles both token shapes: flat ``tokens.prompt`` (newer) and
    ``tokens.by_type.prompt`` (pre-cedb157).
    """
    day = (record or {}).get("date")
    if not day:
        return []
    tok = record.get("tokens") or {}
    by_type = tok.get("by_type") or {}
    inp = int(tok.get("prompt") or by_type.get("prompt") or 0)
    out = int(tok.get("completion") or by_type.get("completion") or 0)
    if inp <= 0 and out <= 0:
        return []
    row = _base_row(SOURCE_KPI, day, "", "", "fleet")
    row["input_tokens"] = inp
    row["output_tokens"] = out
    return [row]


def rows_from_kpi_file(path: str, *, before_day: str) -> list[dict]:
    """Rows for every KPI record strictly before ``before_day`` (the cutover)."""
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if str(record.get("date") or "") >= before_day:
                continue
            rows.extend(rows_from_kpi_record(record))
    return rows


def day_range(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYY-MM-DD`` days from ``start`` to ``end``."""
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    days: list[str] = []
    cursor = first
    while cursor <= last:
        days.append(cursor.isoformat())
        cursor += dt.timedelta(days=1)
    return days


def prometheus_instant_query(base_url: str, query: str, at: float,
                             *, timeout: float = 30.0) -> dict:
    """One ``/api/v1/query`` instant query. Raises on transport or API error.

    A backfill that quietly skips a day it could not fetch would leave an
    invisible hole in the history, so failures propagate to the caller.
    """
    params = urllib.parse.urlencode({"query": query, "time": f"{at:.0f}"})
    url = base_url.rstrip("/") + "/api/v1/query?" + params
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(
            f"prometheus query failed: {payload.get('error') or payload.get('status')}"
        )
    return payload


def prometheus_day_rows(base_url: str, day: str, *, query: str = PROM_QUERY,
                        timeout: float = 30.0) -> list[dict]:
    """Rows for one whole UTC day.

    The instant query is evaluated at 00:00 UTC of the *following* day so the
    ``[1d]`` window covers exactly ``day``.
    """
    at = dt.datetime.combine(
        dt.date.fromisoformat(day) + dt.timedelta(days=1),
        dt.time(0, 0), tzinfo=dt.timezone.utc,
    ).timestamp()
    payload = prometheus_instant_query(base_url, query, at, timeout=timeout)
    return rows_from_prometheus(payload, day)


_INSERT_COLUMNS = (
    "request_id", "ts", "day", "principal", "client", "alias", "model",
    "provider", "outcome", "streamed", "duration_ms", "ttft_ms",
    "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
    "usage_source", "reasoning_source", "synthetic", "message_count",
    "system_hash", "prefix_hash", "tool_count", "temperature", "max_tokens",
    "confidentiality", "fell_back",
)


def _row_is_legal(row: dict) -> bool:
    """Pre-validate the of-which invariant the schema also enforces.

    This matters because ``INSERT OR IGNORE`` makes SQLite ignore EVERY
    constraint failure, not just the primary-key conflict we want to ignore. A
    row violating ``CHECK (reasoning_tokens <= output_tokens)`` would otherwise
    vanish with no error and be indistinguishable from an idempotent skip --
    silent undercounting, which is the failure mode this whole store exists to
    end. Validate first so a rejection can be reported.
    """
    return int(row.get("reasoning_tokens") or 0) <= int(row.get("output_tokens") or 0)


def backfill(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert rows, skipping ids already present. Returns rows inserted.

    Raises ValueError on a schema-illegal row rather than letting OR IGNORE
    swallow it: a backfill that silently drops history is worse than one that
    stops and says so.
    """
    inserted = 0
    sql = (
        "INSERT OR IGNORE INTO requests (" + ", ".join(_INSERT_COLUMNS) + ") "
        "VALUES (" + ", ".join("?" for _ in _INSERT_COLUMNS) + ")"
    )
    for row in rows:
        if not _row_is_legal(row):
            raise ValueError(
                f"illegal backfill row for {row.get('day')}/{row.get('model')}: "
                f"reasoning_tokens exceeds output_tokens"
            )
        cur = conn.execute(sql, tuple(row.get(c) for c in _INSERT_COLUMNS))
        inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    return inserted
