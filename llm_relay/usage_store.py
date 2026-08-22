"""Durable per-request usage rows in SQLite — the system of record.

Why this exists: Prometheus counters reset when the relay restarts,
``increase()`` extrapolates rather than counts, and retention is 30 days. None
of that is acceptable for usage accounting, so exact integers are written here
at the source and Prometheus keeps only its operational duties.

Best-effort by design, exactly like :mod:`llm_relay.audit`: recording must
never raise into the request path. Writes go through a bounded queue drained by
one daemon thread — a single writer, so there is no lock contention and no
delete-then-insert race of the kind the portal's two rollup writers had.
A saturated queue sheds load and counts the loss rather than blocking a request.
"""
from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time

SCHEMA_VERSION = 2

_COLUMNS = (
    "request_id", "ts", "day", "principal", "client", "alias", "model",
    "provider", "outcome", "streamed", "duration_ms", "ttft_ms",
    "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
    "usage_source", "reasoning_source", "synthetic", "request_count",
    "message_count", "system_hash", "prefix_hash", "tool_count", "temperature",
    "max_tokens", "confidentiality", "fell_back",
)

_REQUIRED = ("request_id", "ts", "day", "model", "usage_source")

# Columns a writer may legitimately omit, and what to store when it does.
#
# The live path records one row per request and builds no ``request_count`` at
# all. Binding that missing key straight through would offer NULL to a NOT NULL
# column, and ``INSERT OR IGNORE`` ignores *every* constraint failure -- so the
# relay would record nothing, silently, for every real request. Defaulting here
# means the store defends itself against any writer, present or future, rather
# than depending on each one remembering the column.
_DEFAULTS = {"request_count": 1}


def row_values(row: dict, columns: tuple = _COLUMNS) -> tuple:
    """Bind ``row`` to ``columns``, filling omitted ones from :data:`_DEFAULTS`.

    Shared with :mod:`llm_relay.usage_backfill` so live and backfill inserts
    cannot disagree about what an absent column means.
    """
    out = []
    for column in columns:
        value = row.get(column)
        out.append(_DEFAULTS.get(column) if value is None else value)
    return tuple(out)

_DDL = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        TEXT PRIMARY KEY,
    ts                REAL NOT NULL,
    day               TEXT NOT NULL,
    principal         TEXT NOT NULL DEFAULT '',
    client            TEXT NOT NULL DEFAULT '',
    alias             TEXT,
    model             TEXT NOT NULL,
    provider          TEXT NOT NULL DEFAULT '',
    outcome           TEXT NOT NULL DEFAULT '',
    streamed          INTEGER NOT NULL DEFAULT 0,
    duration_ms       INTEGER,
    ttft_ms           INTEGER,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    usage_source      TEXT NOT NULL,
    reasoning_source  TEXT NOT NULL DEFAULT 'none',
    synthetic         INTEGER NOT NULL DEFAULT 0,
    -- How many requests this row accounts for. A live row is exactly one, which
    -- is the default; a synthetic backfill row is a whole day's aggregate and
    -- carries the real count, so request totals sum request_count and never
    -- COUNT(*). Deliberately no CHECK: proportional splitting can floor a small
    -- share to 0, and a CHECK would make INSERT OR IGNORE drop that row without
    -- a trace -- an undercount worse than the zero it was guarding against.
    request_count     INTEGER NOT NULL DEFAULT 1,
    message_count     INTEGER,
    system_hash       TEXT,
    prefix_hash       TEXT,
    tool_count        INTEGER,
    temperature       REAL,
    max_tokens        INTEGER,
    confidentiality   TEXT,
    fell_back         INTEGER,
    -- Reasoning is an of-which subset of output; the invariant is enforced by
    -- the database so no future writer can quietly violate it.
    CHECK (reasoning_tokens <= output_tokens)
);
CREATE INDEX IF NOT EXISTS requests_day       ON requests(day);
CREATE INDEX IF NOT EXISTS requests_principal ON requests(principal, ts);
CREATE INDEX IF NOT EXISTS requests_model_day ON requests(model, day);
CREATE INDEX IF NOT EXISTS requests_prefix    ON requests(prefix_hash);
"""


# Columns added after v1 shipped, as (name, column definition).
#
# ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so a database
# created by an older schema never gains a new column from _DDL alone. Every
# addition must be listed here too, must be additive, and -- being NOT NULL --
# must carry a DEFAULT so SQLite can supply a value for the rows already on disk.
_MIGRATIONS = (
    ("request_count", "request_count INTEGER NOT NULL DEFAULT 1"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing ``requests`` table up to the current column set.

    ``ALTER TABLE ADD COLUMN`` raises when the column is already present, so
    each step is gated on what the table actually has rather than on a version
    number that a hand-edited or half-migrated file could misreport.
    """
    have = {r[1] for r in conn.execute("PRAGMA table_info(requests)").fetchall()}
    for column, definition in _MIGRATIONS:
        if column not in have:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {definition}")


def open_db(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the usage database with schema applied.

    WAL so a reader never blocks the writer. ``check_same_thread=False`` because
    the writer thread owns the connection while tests read from the main thread.

    Safe to call against a database written by an older schema: missing columns
    are added in place, which is how the live store on the gateway upgrades
    without a dump-and-reload of the history it is the only copy of.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_DDL)
    _migrate(conn)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


_INSERT = (
    "INSERT OR IGNORE INTO requests (" + ", ".join(_COLUMNS) + ") VALUES ("
    + ", ".join("?" for _ in _COLUMNS) + ")"
)


class UsageStore:
    """Queue-backed writer for usage rows. One instance owns one database."""

    def __init__(self, path: str, *, maxsize: int = 10000, autostart: bool = True):
        self.path = path
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        if autostart:
            self._start()

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._drain, name="usage-store-writer", daemon=True
        )
        self._thread.start()

    def record(self, row: dict) -> None:
        """Enqueue one usage row. Never raises; never blocks."""
        try:
            if self._closed:
                self.dropped += 1
                return
            self._q.put_nowait(row)
        except queue.Full:
            # Shed the oldest so a burst costs old data, not the request path.
            try:
                self._q.get_nowait()
                self._q.put_nowait(row)
            except Exception:
                pass
            self.dropped += 1
        except Exception:
            self.dropped += 1

    def _write(self, rows: list) -> None:
        if self._conn is None:
            self._conn = open_db(self.path)
        params = []
        for row in rows:
            if not all(row.get(k) is not None for k in _REQUIRED):
                self.dropped += 1
                continue
            # The of-which invariant is checked here as well as in the schema.
            # INSERT OR IGNORE — which we need so a re-run backfill dedupes on
            # request_id — makes SQLite ignore *every* constraint failure, so a
            # row with reasoning > output would vanish with no error and no
            # count. Silent loss in the system of record is the exact failure
            # this store exists to end, so catch it here and count it.
            if (row.get("reasoning_tokens") or 0) > (row.get("output_tokens") or 0):
                self.dropped += 1
                continue
            params.append(row_values(row))
        if not params:
            return
        try:
            self._conn.executemany(_INSERT, params)
        except sqlite3.Error:
            # A bad batch must not poison the writer; retry rows individually so
            # one malformed row cannot discard its well-formed neighbours.
            for p in params:
                try:
                    self._conn.execute(_INSERT, p)
                except sqlite3.Error:
                    self.dropped += 1

    def _drain(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            batch = [item]
            while len(batch) < 200:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                with self._lock:
                    self._write(batch)
            except Exception:
                self.dropped += len(batch)

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue is drained. For tests and shutdown only."""
        if self._thread is None:  # autostart=False: drain synchronously
            # No writer thread exists, so waiting for the queue to empty on its
            # own would burn the whole timeout for nothing. Drain it here.
            rows = []
            while True:
                try:
                    rows.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if rows:
                try:
                    with self._lock:
                        self._write(rows)
                except Exception:
                    self.dropped += len(rows)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty():
                break
            time.sleep(0.01)
        time.sleep(0.05)  # let an in-flight batch commit

    def close(self) -> None:
        self._closed = True
        try:
            self.flush(timeout=1.0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


_store: UsageStore | None = None
_store_lock = threading.Lock()


def get_store() -> "UsageStore | None":
    """Process-wide store from ``LLM_RELAY_USAGE_DB``; None when unconfigured.

    Unset means the feature is off, which is how this ships: the store can be
    enabled, relocated, or disabled without a code change.
    """
    global _store
    path = os.environ.get("LLM_RELAY_USAGE_DB", "").strip()
    if not path:
        return None
    with _store_lock:
        if _store is None or _store.path != path:
            _store = UsageStore(path)
        return _store


def reset_store_for_tests() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
