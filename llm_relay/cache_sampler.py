"""Fleet-level prefix-cache reuse, sampled from each backend's ``/metrics``.

Why this exists
---------------
vLLM's automatic prefix caching is enabled and working across the fleet, but it
publishes reuse as cumulative counters on the backend's own ``/metrics``:

    vllm:prefix_cache_queries_total{engine="0",model_name="qwen3.6-35b"} 2.05165845e+08
    vllm:prefix_cache_hits_total{engine="0",model_name="qwen3.6-35b"}    1.89851904e+08

Those counters are TOKEN-weighted, not request-weighted -- the numerator is
cached prompt tokens that hit and the denominator is prompt tokens queried, so
``hits / queries`` is literally "share of input tokens that avoided prefill"
and drops straight into a cost model with no conversion. Re-lookups by
preempted requests are tracked separately by vLLM (``preempted_queries`` /
``preempted_hits``) and are NOT folded into these counters, so the denominator
is not inflated by preemption churn.

Deliberately NOT per-request attribution
----------------------------------------
Per-request cache attribution is a different mechanism that already exists:
vLLM reports ``usage.prompt_tokens_details.cached_tokens`` when a serve is
started with ``--enable-prompt-tokens-details`` (off by default), llama.cpp
reports the same number plus ``timings.cache_n``, and ``usage_math`` already
reads both into ``requests.cache_read_tokens``. That lane is where per-principal
accounting belongs.

This module is the AGGREGATE lane, and it is worth keeping for two things the
per-request field cannot do:

1. Fleet-level reuse trend and eviction diagnosis per backend and model -- a
   per-request ``cached_tokens`` tells you a miss happened, never why.
2. Coverage for backends where the serve flag is not (or cannot be) enabled;
   turning it on needs a restart of every serve, and one backend sits on
   borrowed hardware behind a tunnel.

Consequently these numbers are NOT reconciled against
``requests.cache_read_tokens`` and are not expected to match it. Different
grain, different population, different clock: the scrape covers everything the
engine did (including traffic that never came through this relay) between two
samples, while the request rows cover exactly the requests the relay saw.
A discrepancy is the expected state, not a bug to be "fixed" by making one
feed the other.

Accounting from a cumulative counter
------------------------------------
The counters count from backend process start, so the only safe way to get a
per-day number is a delta against a persisted cursor. Two failure modes are
handled explicitly:

* **Restart.** A current value LOWER than the last sample means the process
  restarted; the current value IS the delta (tokens since the restart) rather
  than a negative.
* **First sight.** The first observation of a backend covers an unknown span
  that predates sampling, so it seeds the cursor and attributes nothing. Run
  #1 writing zeros is correct, not a bug.

The cursor advance and the daily accumulation happen in one transaction, which
is what makes a re-run idempotent: replaying an unchanged counter yields a zero
delta and adds nothing.

Schema is additive (``CREATE TABLE IF NOT EXISTS`` plus a column backfill), so
pointing this at the live usage database upgrades it in place and leaves the
``requests`` table -- which this module never writes to -- untouched.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .usage_store import open_db

SOURCE = "vllm_prefix_cache"

# What one observation of one (backend, model) did to the books.
OUTCOME_BASELINE = "baseline"          # first sight: cursor seeded, nothing attributed
OUTCOME_COUNTED = "counted"            # a normal forward delta was added
OUTCOME_RESET = "reset"                # backend restarted; post-restart total added
OUTCOME_UNCHANGED = "unchanged"        # counters had not moved
OUTCOME_REJECTED = "rejected"          # delta failed validation; nothing written
OUTCOME_NOT_REPORTED = "not_reported"  # backend exposes no prefix-cache series

# Anthropic's cache pricing, expressed so a consumer can price reuse without
# rediscovering the multipliers: a cache READ bills at 10% of the input rate, a
# cache WRITE at 125%. ``priced_input_tokens`` defaults the write multiplier to
# 1.0 (plain input at list price) because a vLLM prefix-cache MISS is not the
# same commercial event as an explicit Anthropic cache write -- a caller
# modelling that has to opt in and say so.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

DEFAULT_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheCounters:
    """Token-weighted prefix-cache counters: queried tokens and hit tokens."""

    queried: int = 0
    hits: int = 0


@dataclass(frozen=True)
class MetricsReading:
    """What one backend's ``/metrics`` said about prefix caching.

    ``reported`` is False only when the exposition carries no
    ``vllm:prefix_cache_*`` series at all -- the backend is not vLLM, or has
    caching off. That is a different fact from a series reading zero, and the
    two must never be collapsed: "not reported" is unknown, zero is measured.
    """

    reported: bool
    by_model: dict[str, CacheCounters]


@dataclass(frozen=True)
class Backend:
    """One scrape target: a backend root URL and the models it is known to serve.

    ``models`` comes from config and is used for two things only -- keying a
    "not reported" row, and attributing an unlabelled metric when the backend
    serves exactly one model.
    """

    key: str
    provider: str
    base_url: str
    models: tuple[str, ...] = ()


@dataclass
class SampleResult:
    """Outcome of one pass over the fleet. Every bucket is reported, because a
    sampler that silently skips backends is indistinguishable from one that
    found nothing."""

    day: str
    counted: int = 0
    baselined: int = 0
    resets: int = 0
    unchanged: int = 0
    rejected: int = 0
    queried_tokens: int = 0
    cache_read_tokens: int = 0
    unreachable: list[str] = field(default_factory=list)
    not_reported: list[str] = field(default_factory=list)
    unattributed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing the Prometheus exposition
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"^vllm:prefix_cache_(?P<kind>queries|hits)_total"
    r"(?:\{(?P<labels>[^}]*)\})?[ \t]+(?P<value>\S+)"
)
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _counter_value(raw: str) -> int | None:
    """Parse one counter sample. ``None`` for anything not a finite count.

    vLLM emits scientific notation (``1.89851904e+08``), and Prometheus permits
    ``NaN`` / ``+Inf``. A non-finite or negative counter is not a number of
    tokens, so it is dropped rather than coerced into a misleading zero.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return int(value)


def parse_prefix_cache_metrics(text: str) -> MetricsReading:
    """Extract per-model prefix-cache counters from a Prometheus exposition.

    Keys on the ``model_name`` label. A series with no labels is returned under
    the empty-string key so the caller can decide whether it is attributable
    (see ``sample_backends``) instead of this function guessing.
    """
    reported = False
    queried: dict[str, int] = {}
    hits: dict[str, int] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if match is None:
            continue
        reported = True  # the series exists even if this sample is unusable
        value = _counter_value(match.group("value"))
        if value is None:
            continue
        labels = dict(_LABEL_RE.findall(match.group("labels") or ""))
        model = labels.get("model_name", "")
        target = queried if match.group("kind") == "queries" else hits
        # Several engines on one backend expose the same model_name; the fleet
        # total for that model is their sum.
        target[model] = target.get(model, 0) + value

    by_model = {
        model: CacheCounters(queried.get(model, 0), hits.get(model, 0))
        for model in sorted(set(queried) | set(hits))
    }
    return MetricsReading(reported=reported, by_model=by_model)


# ---------------------------------------------------------------------------
# Discovering backends from the relay's own config
# ---------------------------------------------------------------------------

def backends_from_config(config) -> list[Backend]:
    """Scrape targets derived from providers.yaml + models.yaml.

    Mirrors the (provider, port, path) grouping ``api.app`` uses to register
    discovery clients, so the sampler sees exactly the backends the relay
    routes to -- addresses are never hardcoded here, because the fleet's
    addresses change. Disabled providers are skipped; ``discover_ports`` with
    no configured model are included (they serve ad-hoc models that still
    consume cache).
    """
    backends: dict[str, Backend] = {}
    for provider_name, provider in (getattr(config, "providers", {}) or {}).items():
        if not getattr(provider, "enabled", True):
            continue
        base = (provider.base_url or "").rstrip("/")
        models_config = getattr(getattr(config, "models", None), "models", {}) or {}
        groups: dict[tuple[int | None, str], list[str]] = {}
        for name, model in models_config.items():
            if model.provider != provider_name:
                continue
            groups.setdefault((model.port, model.path or ""), []).append(name)
        if not groups:
            groups[(None, "")] = []
        for (port, path), names in groups.items():
            url = base
            if port:
                url = f"{url}:{port}"
            if path:
                url = f"{url}/{path.lstrip('/')}"
            key = _backend_key(provider_name, port, path)
            backends[key] = Backend(
                key=key, provider=provider_name, base_url=url,
                models=tuple(sorted(names)),
            )
        for port in getattr(provider, "discover_ports", []) or []:
            key = _backend_key(provider_name, port, "")
            if key in backends:
                continue  # a configured model already covers this port
            backends[key] = Backend(
                key=key, provider=provider_name, base_url=f"{base}:{port}",
            )
    return [backends[k] for k in sorted(backends)]


def _backend_key(provider_name: str, port: int | None, path: str) -> str:
    """Same key shape ``routing.keys.compose_backend_key`` produces.

    Imported lazily so this module stays importable in a CLI that has not
    loaded the routing stack (and its heavier dependencies).
    """
    from .routing.keys import compose_backend_key

    return compose_backend_key(provider_name, port, path)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def metrics_url(base_url: str) -> str:
    """``/metrics`` hangs off the backend root, not off the ``/v1`` prefix."""
    return (base_url or "").rstrip("/") + "/metrics"


def _upstream_bearer() -> str | None:
    """Homelab convention: one shared upstream key for every backend.

    Same two env vars ``discovery.endpoint`` reads; duplicated rather than
    imported so the sampler does not pull the httpx-backed discovery stack into
    a stdlib-only CLI.
    """
    return os.environ.get("LLM_RELAY_UPSTREAM_API_KEY") or os.environ.get("LLM_API_KEY")


def fetch_metrics(base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """GET one backend's ``/metrics``. Raises on any transport failure.

    urllib rather than httpx on purpose: this runs from a shared venv outside
    the release, where a new third-party import fails only at exec time in
    production. Always bounded by ``timeout`` -- an unbounded scrape of a wedged
    backend would hang the whole periodic sample.
    """
    request = urllib.request.Request(metrics_url(base_url))
    bearer = _upstream_bearer()
    if bearer:
        request.add_header("Authorization", f"Bearer {bearer}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS cache_daily (
    sample_id         TEXT PRIMARY KEY,
    day               TEXT NOT NULL,
    model             TEXT NOT NULL,
    queried_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    -- 1: a backend reported prefix-cache counters for this model. 0: it was
    -- reachable but exposes no such series, so reuse is UNKNOWN, not zero.
    reported          INTEGER NOT NULL DEFAULT 1,
    resets            INTEGER NOT NULL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'vllm_prefix_cache',
    updated_ts        REAL NOT NULL DEFAULT 0,
    -- Hits are an of-which subset of queries, enforced by the database so no
    -- future writer can quietly produce a hit rate above 100%.
    CHECK (cache_read_tokens <= queried_tokens),
    CHECK (queried_tokens >= 0),
    CHECK (cache_read_tokens >= 0),
    CHECK (reported IN (0, 1)),
    -- Redundant with the deterministic sample_id, but makes the grain
    -- structural: one row per (day, model), no finer.
    UNIQUE (day, model)
);
CREATE INDEX IF NOT EXISTS cache_daily_day ON cache_daily(day);

CREATE TABLE IF NOT EXISTS cache_cursor (
    backend     TEXT NOT NULL,
    model       TEXT NOT NULL,
    queried     INTEGER NOT NULL,
    hits        INTEGER NOT NULL,
    resets      INTEGER NOT NULL DEFAULT 0,
    observed_ts REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (backend, model)
);
"""

# Columns added after the first release, as {table: {column: DDL fragment}}.
# Applied by ALTER TABLE ADD COLUMN so an existing live database upgrades in
# place instead of needing a migration step.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {}


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create this module's tables if absent, then backfill any added column.

    Purely additive and safe to run on every open: it never touches
    ``requests`` and never drops or rewrites a column.
    """
    conn.executescript(_DDL)
    for table, columns in _ADDED_COLUMNS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for column, ddl in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def tables_exist(conn: sqlite3.Connection) -> bool:
    """Whether this module's tables have been created in *conn*.

    A reader has to ask. These tables are created by ``ensure_schema`` -- i.e.
    by the sampler -- and NOT by ``usage_store.open_db``, so a usage store that
    has never been sampled holds request rows and no ``cache_daily`` at all.
    That is the state of every fresh deployment. Creating the table from a read
    path would be a write side effect on a caller whose only honest answer is
    "nothing has been sampled here yet".
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cache_daily'"
    ).fetchone()
    return row is not None


def open_cache_db(path: str) -> sqlite3.Connection:
    """Open the usage database with both the request schema and this one.

    Reuses ``usage_store.open_db`` so the WAL and synchronous pragmas cannot
    drift between the two writers, then layers this module's tables on top.
    """
    conn = open_db(path)
    ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Deltas and writes
# ---------------------------------------------------------------------------

def daily_id(day: str, model: str, source: str = SOURCE) -> str:
    """Deterministic row id so a re-run updates rather than duplicating."""
    key = "|".join((source, day or "", model or ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def counter_delta(previous: CacheCounters | None,
                  current: CacheCounters) -> tuple[CacheCounters, bool]:
    """Tokens accumulated since ``previous``, and whether a reset was detected.

    Three cases, all of which a naive subtraction gets wrong:

    * ``previous is None`` -- first sight. The counter covers an unknown span
      before sampling began, so attributing it to today would put months of
      reuse on one day. Returns a zero delta; the caller seeds the cursor.
    * either counter LOWER than last time -- the backend process restarted and
      its counters began again at zero. The current value is the delta.
    * otherwise -- a plain forward difference.
    """
    if previous is None:
        return CacheCounters(0, 0), False
    if current.queried < previous.queried or current.hits < previous.hits:
        return CacheCounters(current.queried, current.hits), True
    return CacheCounters(current.queried - previous.queried,
                         current.hits - previous.hits), False


def read_cursor(conn: sqlite3.Connection, backend: str,
                model: str) -> CacheCounters | None:
    row = conn.execute(
        "SELECT queried, hits FROM cache_cursor WHERE backend = ? AND model = ?",
        (backend, model),
    ).fetchone()
    return None if row is None else CacheCounters(int(row[0]), int(row[1]))


def _delta_is_legal(delta: CacheCounters) -> bool:
    """Pre-validate what the schema also checks.

    Upstream counters are not trusted. A delta with more hits than queries
    violates ``CHECK (cache_read_tokens <= queried_tokens)``, and the row is
    created with ``INSERT OR IGNORE`` -- which makes SQLite ignore EVERY
    constraint failure, not just the primary-key conflict we want ignored. An
    illegal row would vanish with no error and be indistinguishable from an
    idempotent skip, so it is caught here and counted instead.
    """
    return 0 <= delta.hits <= delta.queried


class _Transaction:
    """Explicit transaction: the connection is opened in autocommit mode, so
    without this the cursor advance and the daily accumulation could tear apart
    and a crash between them would double-count or lose a delta."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.owned = False

    def __enter__(self):
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
            self.owned = True
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if not self.owned:
            return False
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False


_ENSURE_ROW = (
    "INSERT OR IGNORE INTO cache_daily "
    "(sample_id, day, model, queried_tokens, cache_read_tokens, reported, "
    " resets, source, updated_ts) VALUES (?, ?, ?, 0, 0, 0, 0, ?, ?)"
)

_ADD_DELTA = (
    "UPDATE cache_daily SET "
    "  queried_tokens = queried_tokens + ?, "
    "  cache_read_tokens = cache_read_tokens + ?, "
    "  resets = resets + ?, "
    "  reported = MAX(reported, ?), "
    "  updated_ts = ? "
    "WHERE sample_id = ?"
)


def _apply(conn: sqlite3.Connection, *, day: str, model: str,
           delta: CacheCounters, reported: bool, resets: int, now: float) -> bool:
    """Accumulate one delta onto the (day, model) row. False if it was refused.

    The row is created with zeros and ``reported = 0`` first, then updated, so
    the accumulation is a plain UPDATE whose CHECK violations raise instead of
    being swallowed. ``reported`` only ever ratchets up: a backend that starts
    reporting promotes an existing unknown row, and one that stops cannot
    demote a measured one.
    """
    sample_id = daily_id(day, model)
    conn.execute(_ENSURE_ROW, (sample_id, day, model, SOURCE, now))
    cur = conn.execute(_ADD_DELTA, (
        delta.queried, delta.hits, resets, 1 if reported else 0, now, sample_id,
    ))
    # A rowcount of 0 means the deterministic id did not address the row the
    # UNIQUE(day, model) constraint kept -- an id bug, not an idempotent skip.
    return cur.rowcount == 1


def record_sample(conn: sqlite3.Connection, *, day: str, backend: str, model: str,
                  current: CacheCounters, now: float | None = None) -> str:
    """Fold one cumulative reading into the day's totals. Returns an outcome.

    Cursor advance and daily accumulation share a transaction, which is what
    makes this idempotent: replaying an unchanged counter produces a zero delta
    and adds nothing.
    """
    now = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
    previous = read_cursor(conn, backend, model)
    delta, was_reset = counter_delta(previous, current)
    if not _delta_is_legal(delta):
        return OUTCOME_REJECTED
    try:
        with _Transaction(conn):
            if not _apply(conn, day=day, model=model, delta=delta, reported=True,
                          resets=1 if was_reset else 0, now=now):
                raise sqlite3.IntegrityError(
                    f"cache_daily row for {day}/{model} not addressable by its id"
                )
            conn.execute(
                "INSERT INTO cache_cursor (backend, model, queried, hits, resets, "
                "observed_ts) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(backend, model) DO UPDATE SET "
                "  queried = excluded.queried, hits = excluded.hits, "
                "  resets = cache_cursor.resets + excluded.resets, "
                "  observed_ts = excluded.observed_ts",
                (backend, model, current.queried, current.hits,
                 1 if was_reset else 0, now),
            )
    except sqlite3.Error:
        # The schema refused the accumulation. Nothing was written (the
        # transaction rolled back); report it rather than losing it silently.
        return OUTCOME_REJECTED
    if previous is None:
        return OUTCOME_BASELINE
    if was_reset:
        return OUTCOME_RESET
    if delta.queried == 0 and delta.hits == 0:
        return OUTCOME_UNCHANGED
    return OUTCOME_COUNTED


def record_not_reported(conn: sqlite3.Connection, *, day: str, model: str,
                        now: float | None = None) -> str:
    """Record that a reachable backend exposes no prefix-cache series.

    Writes a zero row flagged ``reported = 0`` so "we asked and it does not
    say" is a stored fact, distinguishable from a measured zero. Never demotes
    a row another backend already reported.
    """
    now = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
    try:
        with _Transaction(conn):
            _apply(conn, day=day, model=model, delta=CacheCounters(0, 0),
                   reported=False, resets=0, now=now)
    except sqlite3.Error:
        return OUTCOME_REJECTED
    return OUTCOME_NOT_REPORTED


# ---------------------------------------------------------------------------
# Sampling the fleet
# ---------------------------------------------------------------------------

def utc_day(now: float | None = None) -> str:
    when = (dt.datetime.now(dt.timezone.utc) if now is None
            else dt.datetime.fromtimestamp(now, dt.timezone.utc))
    return when.date().isoformat()


def sample_backends(conn: sqlite3.Connection, backends, *, day: str | None = None,
                    timeout: float = DEFAULT_TIMEOUT, fetcher=None,
                    now: float | None = None) -> SampleResult:
    """Scrape every backend once and fold the results into ``day``.

    One unreachable backend must never cost the others their sample, so every
    fetch and every parse is isolated; failures are collected and reported.
    Deltas are attributed to the day of the sample, so a delta spanning
    midnight lands on the later day -- bounded by the sampling interval, which
    is why this is meant to run frequently rather than once daily.
    """
    day = day or utc_day(now)
    fetch = fetcher or fetch_metrics
    result = SampleResult(day=day)
    for backend in backends:
        try:
            text = fetch(backend.base_url, timeout=timeout)
            reading = parse_prefix_cache_metrics(text)
        except Exception:
            # Transport error, timeout, garbage body -- any of them mean we did
            # not learn anything about this backend. Recording a zero would be
            # a lie; recording nothing and saying so is not.
            result.unreachable.append(backend.key)
            continue
        if not reading.reported:
            result.not_reported.append(backend.key)
            for model in backend.models:
                record_not_reported(conn, day=day, model=model, now=now)
            continue
        for model, counters in reading.by_model.items():
            resolved = model
            if not resolved:
                # An unlabelled series is only attributable when the backend
                # serves exactly one model; guessing otherwise would put one
                # model's reuse on another's bill.
                if len(backend.models) != 1:
                    result.unattributed.append(backend.key)
                    continue
                resolved = backend.models[0]
            before = read_cursor(conn, backend.key, resolved)
            outcome = record_sample(conn, day=day, backend=backend.key,
                                    model=resolved, current=counters, now=now)
            if outcome == OUTCOME_REJECTED:
                result.rejected += 1
                continue
            if outcome == OUTCOME_BASELINE:
                result.baselined += 1
                continue
            if outcome == OUTCOME_UNCHANGED:
                result.unchanged += 1
                continue
            delta, _ = counter_delta(before, counters)
            result.counted += 1
            result.queried_tokens += delta.queried
            result.cache_read_tokens += delta.hits
            if outcome == OUTCOME_RESET:
                result.resets += 1
    return result


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

def _hit_rate(queried: int, hits: int) -> float | None:
    """Share of queried input tokens served from cache, or None when unknown.

    Nothing queried means no rate exists. Returning 0.0 there would read as
    "no reuse", which is the exact conflation this module exists to end.
    """
    return (hits / queried) if queried > 0 else None


def cache_rollup(conn: sqlite3.Connection, start_day: str,
                 end_day: str) -> list[dict]:
    """Per (day, model) cache totals and hit rate in an inclusive window."""
    cur = conn.execute(
        "SELECT day, model, queried_tokens, cache_read_tokens, reported, "
        "       resets, source, updated_ts "
        "FROM cache_daily WHERE day >= ? AND day <= ? "
        "ORDER BY day, model",
        (start_day, end_day),
    )
    return [
        {
            "day": r[0], "model": r[1],
            "queried_tokens": int(r[2] or 0),
            "cache_read_tokens": int(r[3] or 0),
            "hit_rate": _hit_rate(int(r[2] or 0), int(r[3] or 0)),
            "reported": bool(r[4]),
            "resets": int(r[5] or 0),
            "source": r[6],
            "updated_ts": r[7],
        }
        for r in cur.fetchall()
    ]


def cache_by_model(conn: sqlite3.Connection, start_day: str,
                   end_day: str) -> list[dict]:
    """Per-model cache totals over a window, with the hit rate to price from.

    ``reported`` is the OR across the window: a model measured on any day is
    reported, even if some backend serving it never exposed the series.
    """
    cur = conn.execute(
        "SELECT model, SUM(queried_tokens), SUM(cache_read_tokens), "
        "       MAX(reported), SUM(resets), COUNT(DISTINCT day) "
        "FROM cache_daily WHERE day >= ? AND day <= ? "
        "GROUP BY model ORDER BY model",
        (start_day, end_day),
    )
    return [
        {
            "model": r[0],
            "queried_tokens": int(r[1] or 0),
            "cache_read_tokens": int(r[2] or 0),
            "hit_rate": _hit_rate(int(r[1] or 0), int(r[2] or 0)),
            "reported": bool(r[3]),
            "resets": int(r[4] or 0),
            "days": int(r[5] or 0),
        }
        for r in cur.fetchall()
    ]


def priced_input_tokens(queried_tokens: int, cache_read_tokens: int, *,
                        read_multiplier: float = CACHE_READ_MULTIPLIER,
                        write_multiplier: float = 1.0) -> float:
    """Input tokens re-expressed in list-price-equivalent units.

    A cache read bills at ``read_multiplier`` of the input rate (Anthropic:
    0.10), the remainder at ``write_multiplier`` (1.0 = plain input; pass
    ``CACHE_WRITE_MULTIPLIER`` to model an explicit cache write at 1.25). Reads
    are clamped to the queried total so an inconsistent upstream pair cannot
    produce a negative fresh-token count.
    """
    queried = max(int(queried_tokens or 0), 0)
    reads = min(max(int(cache_read_tokens or 0), 0), queried)
    return (queried - reads) * write_multiplier + reads * read_multiplier
