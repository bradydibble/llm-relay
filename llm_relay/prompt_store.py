"""Content-addressed prompt and completion storage.

Each distinct message is stored ONCE, keyed by the sha256 of its redacted text,
with a join table mapping (request, ordinal) -> message. Agent traffic resends
the whole conversation every turn, so this collapses a 100-turn session from
~100 copies of its history to one copy per distinct message. The resulting
dedup ratio is also the most useful optimization signal available: it measures
the prompt-cache opportunity directly.

Separate database from usage.db on purpose: different sensitivity, different
retention, different backup posture, and a problem in one cannot corrupt the
other. ``prompt_requests`` duplicates five small columns from usage.db so this
file is self-sufficient for filtering and pruning without a cross-database
ATTACH.

Best-effort, like audit.py and usage_store.py: never raises into a request.
"""
from __future__ import annotations

import hashlib
import os
import queue
import re
import sqlite3
import threading
import time
import zlib

from .redaction import redact

SCHEMA_VERSION = 1

# Codec resolution happens once, at import. ``compression.zstd`` is Python
# 3.14+ and the relay runs 3.11, so this is zlib unless the ``zstandard``
# package happens to be installed in the venv. The codec is recorded per blob
# so adding zstd later needs no migration and no rewrite of existing rows.
try:  # pragma: no cover - depends on the deployed venv
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None

_CODEC = "zstd" if _zstd is not None else "zlib"

# Bounds on a single search: enough terms to be expressive, few enough that one
# query cannot turn into an unbounded FTS scan.
_MAX_TERMS = 16
_MAX_LIMIT = 500

_WORD = re.compile(r"\w+")
_WHITESPACE = re.compile(r"\s+")

_DDL = """
CREATE TABLE IF NOT EXISTS prompt_requests (
    request_id TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    day        TEXT NOT NULL,
    principal  TEXT NOT NULL DEFAULT '',
    client     TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS prompt_requests_day  ON prompt_requests(day);
CREATE INDEX IF NOT EXISTS prompt_requests_prin ON prompt_requests(principal, ts);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,
    hash       TEXT NOT NULL UNIQUE,
    role       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    content    BLOB NOT NULL,
    codec      TEXT NOT NULL,
    redacted   INTEGER NOT NULL DEFAULT 0,
    first_seen REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS request_messages (
    request_id TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    PRIMARY KEY (request_id, ordinal)
);
CREATE INDEX IF NOT EXISTS request_messages_msg ON request_messages(message_id);

CREATE TABLE IF NOT EXISTS completions (
    request_id TEXT PRIMARY KEY,
    content    BLOB,
    reasoning  BLOB,
    codec      TEXT NOT NULL,
    redacted   INTEGER NOT NULL DEFAULT 0
);

-- Contentless FTS5: stores terms only, not a second plaintext copy of every
-- message. snippet() is unavailable as a result, so snippets are built in
-- application code from the decompressed blob.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='');
"""


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #

def compress(text: str) -> tuple[bytes, str]:
    """Return ``(blob, codec)``. The codec travels with the blob, per row."""
    data = (text or "").encode("utf-8", "replace")
    if _zstd is not None:
        try:
            return _zstd.ZstdCompressor(level=3).compress(data), "zstd"
        except Exception:
            pass  # fall through to zlib rather than lose the content
    return zlib.compress(data, 6), "zlib"


def decompress(blob, codec: str) -> str:
    """Inverse of :func:`compress`.

    Raises ``ValueError`` for a codec this interpreter cannot read, rather than
    returning empty text — an admin reading the archive must not be shown a
    blank message that is actually an unreadable one.
    """
    if not blob:
        return ""
    raw = bytes(blob)
    if codec == "zlib":
        return zlib.decompress(raw).decode("utf-8", "replace")
    if codec == "zstd":
        if _zstd is None:
            raise ValueError("blob is zstd but the zstandard package is absent")
        return _zstd.ZstdDecompressor().decompress(raw).decode("utf-8", "replace")
    raise ValueError("unknown codec %r" % (codec,))


def _readable(blob, codec: str) -> str:
    """Decompress for display, substituting a visible marker on failure."""
    try:
        return decompress(blob, codec)
    except Exception as exc:
        return "[unreadable blob: %s]" % (exc,)


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #

def open_db(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the prompt database with schema applied.

    WAL so a reader never blocks the writer. ``check_same_thread=False`` because
    the writer thread owns the connection while readers use their own.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, mode=0o700, exist_ok=True)
        try:
            os.chmod(parent, 0o700)
        except OSError:
            pass
    # 0600 / 0700, explicitly. Nothing in the shipped units sets UMask, so the
    # service default (0022) would create these 0644 -- and this file holds
    # per-user token history (and, in prompts.db, coworkers' actual
    # conversations). Production is currently saved only by a hand-set mode on
    # the parent directory, which is not something this code can rely on: the
    # DB path is deliberately relocatable. auth.py already does this for its
    # key-hash file. Applied on every open so an existing loose file is tightened
    # rather than left as found.
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # A store we cannot tighten is still better than no store; the unit's
        # UMask and the parent mode remain as defence.
        pass
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_DDL)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


def _as_text(content) -> str:
    """Coerce a message ``content`` to text without ever raising.

    ``content`` is a string for ordinary chat, but the OpenAI schema also allows
    a list of parts for multimodal input. Text parts are kept and joined; a
    non-text part (an image, an audio blob, a base64 data URL) is reduced to a
    ``[type]`` marker rather than archived, because this store exists to hold
    conversation text and must not become a binary attachment dump.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append("[%s]" % (part.get("type") or "part",))
            else:
                parts.append("[part]")
        return "\n".join(parts)
    try:
        return str(content)
    except Exception:
        return ""


def _store_message(conn: sqlite3.Connection, role: str, text: str, ts: float) -> int:
    """Redact, hash, store-if-new, index-if-new. Returns the message id.

    The hash covers ``role`` as well as the redacted text: identical text under
    two different roles is two different messages, and collapsing them would
    report the wrong role and break role filtering. Conversation resends repeat
    both, so dedup is unaffected.
    """
    clean, hit = redact(text)
    digest = hashlib.sha256(
        ("%s\x00%s" % (role, clean)).encode("utf-8", "replace")
    ).hexdigest()
    row = conn.execute("SELECT id FROM messages WHERE hash = ?", (digest,)).fetchone()
    if row is not None:
        return int(row[0])
    blob, codec = compress(clean)
    try:
        cur = conn.execute(
            "INSERT INTO messages (hash, role, bytes, content, codec, redacted,"
            " first_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (digest, role, len(clean.encode("utf-8", "replace")), blob, codec,
             1 if hit else 0, ts),
        )
    except sqlite3.IntegrityError:
        # Another process inserted the same content between the SELECT and the
        # INSERT. Adopt its row instead of failing the whole request.
        row = conn.execute(
            "SELECT id FROM messages WHERE hash = ?", (digest,)).fetchone()
        if row is None:
            raise
        return int(row[0])
    message_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO messages_fts (rowid, text) VALUES (?, ?)", (message_id, clean))
    return message_id


def _reject_reason(entry) -> str | None:
    """Why this entry cannot be stored, or None if it can.

    Explicit validation, deliberately not delegated to ``INSERT OR IGNORE``:
    ``OR IGNORE`` makes SQLite ignore *every* constraint failure, so a bad row
    disappears with no error and no counter. Silent loss is the one outcome
    this store must never produce -- "we stored nothing" must never be
    indistinguishable from "there was nothing to store".
    """
    if not isinstance(entry, dict):
        return "not a dict"
    if not str(entry.get("request_id") or "").strip():
        return "missing request_id"
    if entry.get("ts") is None:
        return "missing ts"
    try:
        float(entry["ts"])
    except (TypeError, ValueError):
        return "unparseable ts"
    if not str(entry.get("day") or "").strip():
        return "missing day"
    messages = entry.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no messages"
    return None


def _write_entry(conn: sqlite3.Connection, entry: dict) -> int:
    """Write one request, its messages, its links and its completion.

    Returns the number of individual messages skipped as unusable. The caller
    runs this inside a transaction so a request never half-lands.
    """
    request_id = str(entry["request_id"]).strip()
    ts = float(entry["ts"])
    # Every column is validated above, so the PRIMARY KEY conflict of a
    # re-delivered request is the only failure OR IGNORE can absorb here.
    conn.execute(
        "INSERT OR IGNORE INTO prompt_requests"
        " (request_id, ts, day, principal, client, model) VALUES (?, ?, ?, ?, ?, ?)",
        (request_id, ts, str(entry["day"]).strip(),
         str(entry.get("principal") or ""), str(entry.get("client") or ""),
         str(entry.get("model") or "")),
    )
    skipped = 0
    for ordinal, message in enumerate(entry["messages"]):
        if not isinstance(message, dict):
            skipped += 1
            continue
        message_id = _store_message(
            conn, str(message.get("role") or ""), _as_text(message.get("content")), ts)
        conn.execute(
            "INSERT OR IGNORE INTO request_messages (request_id, ordinal, message_id)"
            " VALUES (?, ?, ?)", (request_id, ordinal, message_id))

    completion, c_hit = redact(_as_text(entry.get("completion")))
    reasoning, r_hit = redact(_as_text(entry.get("reasoning")))
    c_blob = compress(completion)[0] if completion else None
    r_blob = compress(reasoning)[0] if reasoning else None
    conn.execute(
        "INSERT OR IGNORE INTO completions (request_id, content, reasoning, codec,"
        " redacted) VALUES (?, ?, ?, ?, ?)",
        (request_id, c_blob, r_blob, _CODEC, 1 if (c_hit or r_hit) else 0),
    )
    return skipped


class PromptStore:
    """Queue-backed writer for prompt content. One instance owns one database.

    Same shape as :class:`llm_relay.usage_store.UsageStore`: a bounded queue
    drained by one daemon thread, so there is a single writer and a saturated
    queue sheds load and counts the loss instead of blocking a request.
    """

    def __init__(self, path: str, *, maxsize: int = 2000, autostart: bool = True):
        self.path = path
        self.dropped = 0            # whole requests not stored
        self.dropped_messages = 0   # individual messages skipped within a request
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        if autostart:
            self._start()

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._drain, name="prompt-store-writer", daemon=True
        )
        self._thread.start()

    def record(self, entry: dict) -> None:
        """Enqueue one request's content. Never raises; never blocks.

        Shape validation happens here rather than only in the writer thread, so
        a rejected entry is counted before ``record`` returns and never burns a
        queue slot.
        """
        try:
            if self._closed:
                self.dropped += 1
                return
            if _reject_reason(entry) is not None:
                self.dropped += 1
                return
            self._q.put_nowait(entry)
        except queue.Full:
            # Shed the oldest so a burst costs old content, not the request path.
            try:
                self._q.get_nowait()
                self._q.put_nowait(entry)
            except Exception:
                pass
            self.dropped += 1
        except Exception:
            self.dropped += 1

    def _write(self, entries: list) -> None:
        if self._conn is None:
            self._conn = open_db(self.path)
        for entry in entries:
            # Re-checked here: record() is the usual gate, but the writer must
            # not assume it was the only door in.
            if _reject_reason(entry) is not None:
                self.dropped += 1
                continue
            # One transaction per entry. A request either lands whole or not at
            # all -- messages with no links, or links with no request, would be
            # worse than the drop.
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error:
                pass
            try:
                self.dropped_messages += _write_entry(self._conn, entry)
                self._conn.execute("COMMIT")
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
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
            while len(batch) < 50:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                with self._lock:
                    self._write(batch)
            except Exception:
                # open_db itself failed (unwritable path, corrupt file): count
                # the loss and keep the thread alive.
                self.dropped += len(batch)

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue is drained. For tests and shutdown only."""
        if self._thread is None:  # autostart=False: drain synchronously
            entries = []
            while True:
                try:
                    entries.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if entries:
                try:
                    with self._lock:
                        self._write(entries)
                except Exception:
                    self.dropped += len(entries)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty():
                break
            time.sleep(0.01)
        # An empty queue only means the batch was picked up, not that it was
        # committed. The writer holds this lock across a batch, so taking it
        # here waits for the commit -- which is what makes ``dropped`` and the
        # stored rows observable the moment flush() returns. Bounded by the
        # caller's timeout so a wedged writer cannot hang a shutdown.
        remaining = max(0.05, deadline - time.monotonic())
        if self._lock.acquire(timeout=remaining):
            self._lock.release()

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


# --------------------------------------------------------------------------- #
# read path
# --------------------------------------------------------------------------- #

def _fts_query(raw: str) -> str:
    """Reduce arbitrary user input to a safe FTS5 expression.

    Only word characters survive, and each surviving term becomes a quoted
    phrase AND-ed with the rest. That is what keeps a query containing ``"``,
    ``*``, or ``NEAR(`` from becoming a syntax error -- i.e. a 500 -- on an
    admin route.
    """
    if not raw or not isinstance(raw, str):
        return ""
    terms = _WORD.findall(raw)[:_MAX_TERMS]
    if not terms:
        return ""
    return " AND ".join('"%s"' % term for term in terms)


def build_snippet(text: str, query: str, width: int = 240) -> str:
    """A window of ``text`` around the first matching query term.

    Contentless FTS5 cannot provide ``snippet()``, so this is the replacement:
    whitespace is flattened for display, then the earliest term hit anchors the
    window. Falls back to the leading ``width`` characters.
    """
    if not text:
        return ""
    flat = _WHITESPACE.sub(" ", text).strip()
    lowered = flat.lower()
    position = -1
    for term in _WORD.findall(query or ""):
        found = lowered.find(term.lower())
        if found >= 0 and (position < 0 or found < position):
            position = found
    if position < 0:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, position - width // 3)
    end = min(len(flat), start + width)
    snippet = flat[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(flat):
        snippet = snippet + "…"
    return snippet


def search(conn: sqlite3.Connection, query: str, *, principal=None, model=None,
           client=None, start_day=None, end_day=None, role=None,
           limit: int = 50) -> list:
    """Full-text search over stored messages, newest request first.

    One row per (message, request) reference, which is what makes "which
    requests contain this text" answerable for a shared system prompt.
    """
    match = _fts_query(query)
    if not match:
        return []
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, _MAX_LIMIT))

    sql = [
        "SELECT m.id, m.role, m.content, m.codec, m.redacted, m.bytes,",
        "       rm.request_id, rm.ordinal,",
        "       r.ts, r.day, r.principal, r.client, r.model",
        "FROM messages m",
        "JOIN request_messages rm ON rm.message_id = m.id",
        "JOIN prompt_requests r ON r.request_id = rm.request_id",
        "WHERE m.id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?)",
    ]
    params: list = [match]
    for column, value in (("r.principal", principal), ("r.client", client),
                          ("r.model", model), ("m.role", role)):
        if value:
            sql.append("AND %s = ?" % column)
            params.append(str(value))
    if start_day:
        sql.append("AND r.day >= ?")
        params.append(str(start_day))
    if end_day:
        sql.append("AND r.day <= ?")
        params.append(str(end_day))
    sql.append("ORDER BY r.ts DESC, rm.request_id, rm.ordinal LIMIT ?")
    params.append(limit)

    hits = []
    for row in conn.execute(" ".join(sql), params).fetchall():
        (message_id, msg_role, blob, codec, redacted, nbytes, request_id,
         ordinal, ts, day, prin, cli, mdl) = row
        hits.append({
            "message_id": int(message_id),
            "request_id": request_id,
            "ordinal": int(ordinal),
            "role": msg_role,
            "ts": ts,
            "day": day,
            "principal": prin,
            "client": cli,
            "model": mdl,
            "bytes": int(nbytes),
            "redacted": int(redacted),
            "snippet": build_snippet(_readable(blob, codec), query),
        })
    return hits


def read_request(conn: sqlite3.Connection, request_id: str) -> dict:
    """One request in full: its metadata, its messages in order, its output."""
    rid = str(request_id)
    row = conn.execute(
        "SELECT request_id, ts, day, principal, client, model FROM prompt_requests"
        " WHERE request_id = ?", (rid,)).fetchone()
    out = {
        "request_id": rid, "found": row is not None,
        "ts": row[1] if row else None, "day": row[2] if row else "",
        "principal": row[3] if row else "", "client": row[4] if row else "",
        "model": row[5] if row else "",
        "messages": [], "completion": "", "reasoning": "", "redacted": 0,
    }
    for ordinal, role, blob, codec, redacted, message_id in conn.execute(
        "SELECT rm.ordinal, m.role, m.content, m.codec, m.redacted, m.id"
        " FROM request_messages rm JOIN messages m ON m.id = rm.message_id"
        " WHERE rm.request_id = ? ORDER BY rm.ordinal", (rid,)
    ):
        out["messages"].append({
            "ordinal": int(ordinal), "role": role, "message_id": int(message_id),
            "redacted": int(redacted), "content": _readable(blob, codec),
        })
    completion = conn.execute(
        "SELECT content, reasoning, codec, redacted FROM completions"
        " WHERE request_id = ?", (rid,)).fetchone()
    if completion is not None:
        c_blob, r_blob, codec, redacted = completion
        out["completion"] = _readable(c_blob, codec) if c_blob else ""
        out["reasoning"] = _readable(r_blob, codec) if r_blob else ""
        out["redacted"] = int(redacted)
    return out


def stats(conn: sqlite3.Connection, path: str = "") -> dict:
    """Row counts, dedup ratio and on-disk size.

    ``dedup_ratio`` is links per stored message: it *is* the prompt-cache
    opportunity, measured on real traffic rather than estimated.
    """
    stored = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    links = conn.execute("SELECT COUNT(*) FROM request_messages").fetchone()[0]
    requests = conn.execute("SELECT COUNT(*) FROM prompt_requests").fetchone()[0]
    days = conn.execute(
        "SELECT COUNT(DISTINCT day) FROM prompt_requests").fetchone()[0]
    uncompressed = conn.execute(
        "SELECT COALESCE(SUM(bytes), 0) FROM messages").fetchone()[0]
    on_disk = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            on_disk += os.path.getsize(path + suffix)
        except OSError:
            pass
    return {
        "stored_messages": int(stored),
        "message_links": int(links),
        "requests": int(requests),
        "dedup_ratio": round(links / stored, 3) if stored else 1.0,
        "bytes": on_disk,
        "uncompressed_bytes": int(uncompressed),
        "distinct_days": int(days),
        "codec": _CODEC,
    }


def prune(conn: sqlite3.Connection, older_than_day: str) -> dict:
    """Delete requests before ``older_than_day`` and any message left orphaned.

    A message survives while *any* remaining request still references it, which
    is the whole point of content addressing -- a shared system prompt must not
    vanish because the oldest conversation using it aged out.
    """
    day = str(older_than_day or "").strip()
    empty = {"requests_deleted": 0, "messages_deleted": 0, "index_stale": 0}
    if not day:
        return empty
    ids = [r[0] for r in conn.execute(
        "SELECT request_id FROM prompt_requests WHERE day < ?", (day,)).fetchall()]
    if not ids:
        return empty

    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error:
        pass
    try:
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            marks = ", ".join("?" for _ in chunk)
            conn.execute(
                "DELETE FROM request_messages WHERE request_id IN (%s)" % marks, chunk)
            conn.execute(
                "DELETE FROM completions WHERE request_id IN (%s)" % marks, chunk)
            conn.execute(
                "DELETE FROM prompt_requests WHERE request_id IN (%s)" % marks, chunk)

        orphans = conn.execute(
            "SELECT m.id, m.content, m.codec FROM messages m"
            " LEFT JOIN request_messages rm ON rm.message_id = m.id"
            " WHERE rm.message_id IS NULL").fetchall()
        deleted = 0
        stale = 0
        for message_id, blob, codec in orphans:
            # A contentless FTS5 table cannot be DELETEd from (SQLite 3.34 has
            # no contentless_delete option). The documented removal is the
            # 'delete' command replaying the ORIGINAL indexed text, which the
            # blob still holds exactly -- so the index cannot keep surfacing
            # hits for content that no longer exists.
            try:
                text = decompress(blob, codec)
            except Exception:
                text = None
                stale += 1
            if text is not None:
                conn.execute(
                    "INSERT INTO messages_fts (messages_fts, rowid, text)"
                    " VALUES ('delete', ?, ?)", (message_id, text))
            conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            deleted += 1
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    return {"requests_deleted": len(ids), "messages_deleted": deleted,
            "index_stale": stale}


# --------------------------------------------------------------------------- #
# process-wide store
# --------------------------------------------------------------------------- #

_store: PromptStore | None = None
_store_lock = threading.Lock()


def get_store() -> "PromptStore | None":
    """Process-wide store from ``LLM_RELAY_PROMPT_DB``; None when unconfigured.

    Unset means content capture is off, which is how this ships: counts and
    fingerprints can run in production while content capture stays dark until
    it is deliberately switched on.
    """
    global _store
    path = os.environ.get("LLM_RELAY_PROMPT_DB", "").strip()
    if not path:
        return None
    with _store_lock:
        if _store is None or _store.path != path:
            _store = PromptStore(path)
        return _store


def reset_store_for_tests() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
