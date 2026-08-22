"""Content-addressed prompt storage.

The fleet's traffic is ~99% input because agents resend the whole conversation
every turn. Storing each distinct message once instead of once per resend is
what makes indefinite retention viable -- and the dedup ratio IS the
prompt-cache opportunity, measured directly.
"""
from __future__ import annotations

from llm_relay.prompt_store import (
    PromptStore,
    compress,
    decompress,
    open_db,
    prune,
    read_request,
    search,
    stats,
)


def _entry(request_id, messages, *, day="2026-08-20", ts=1787000000.0,
           principal="brady", client="claude-code", model="glm-5.2",
           completion="an answer", reasoning=""):
    return {
        "request_id": request_id, "ts": ts, "day": day, "principal": principal,
        "client": client, "model": model, "messages": messages,
        "completion": completion, "reasoning": reasoning,
    }


def test_compress_roundtrip():
    blob, codec = compress("hello world")
    assert isinstance(blob, bytes)
    assert decompress(blob, codec) == "hello world"


def test_messages_are_stored_and_readable(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("r1", [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "what is a substrate"},
        ]))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        got = read_request(conn, "r1")
        assert [m["role"] for m in got["messages"]] == ["system", "user"]
        assert got["messages"][1]["content"] == "what is a substrate"
        assert got["completion"] == "an answer"
    finally:
        store.close()


def test_a_resent_conversation_stores_each_message_once(tmp_path):
    # THE point of content addressing: three turns of a growing conversation.
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        history = [{"role": "user", "content": "turn one"}]
        store.record(_entry("r1", list(history)))
        history += [{"role": "assistant", "content": "reply one"},
                    {"role": "user", "content": "turn two"}]
        store.record(_entry("r2", list(history)))
        history += [{"role": "assistant", "content": "reply two"},
                    {"role": "user", "content": "turn three"}]
        store.record(_entry("r3", list(history)))
        store.flush()

        conn = open_db(str(tmp_path / "p.db"))
        stored = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM request_messages").fetchone()[0]
        assert stored == 5   # five distinct messages
        assert links == 9    # 1 + 3 + 5 references to them
        s = stats(conn, str(tmp_path / "p.db"))
        assert s["message_links"] == 9
        assert s["stored_messages"] == 5
        assert s["dedup_ratio"] > 1.5
    finally:
        store.close()


def test_secrets_are_redacted_before_storage(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("r1", [
            {"role": "user", "content": "deploy with llmr_aaaabbbbccccddddeeeeffff0000"},
        ]))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        got = read_request(conn, "r1")
        assert "llmr_aaaabbbbccccddddeeeeffff0000" not in got["messages"][0]["content"]
        assert got["messages"][0]["redacted"] == 1
        # And the raw bytes on disk must not contain it either.
        blob = conn.execute("SELECT content FROM messages").fetchone()[0]
        assert b"llmr_aaaabbbbcccc" not in blob
    finally:
        store.close()


def test_search_finds_a_message_by_word(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("r1", [{"role": "user", "content": "how do I restart warewulf"}]))
        store.record(_entry("r2", [{"role": "user", "content": "explain fuzzball substrates"}]))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        hits = search(conn, "warewulf")
        assert len(hits) == 1
        assert hits[0]["request_id"] == "r1"
        assert "warewulf" in hits[0]["snippet"].lower()
    finally:
        store.close()


def test_search_filters_by_principal_and_model_and_day(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("r1", [{"role": "user", "content": "shared word here"}],
                            principal="brady", model="glm-5.2", day="2026-08-20"))
        store.record(_entry("r2", [{"role": "user", "content": "shared word here too"}],
                            principal="jrodriguez", model="ornith-35b", day="2026-08-25"))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        assert len(search(conn, "shared")) == 2
        assert [h["request_id"] for h in search(conn, "shared", principal="brady")] == ["r1"]
        assert [h["request_id"] for h in search(conn, "shared", model="ornith-35b")] == ["r2"]
        assert [h["request_id"] for h in
                search(conn, "shared", start_day="2026-08-24", end_day="2026-08-31")] == ["r2"]
    finally:
        store.close()


def test_search_query_with_fts_metacharacters_does_not_explode(tmp_path):
    # A user typing quotes or an asterisk must not produce a 500.
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("r1", [{"role": "user", "content": 'he said "hello" loudly'}]))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        assert search(conn, '"hello"') is not None
        assert search(conn, "* AND NEAR(") is not None
        assert search(conn, "") == []
    finally:
        store.close()


def test_prune_removes_old_requests_and_orphaned_messages(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record(_entry("old", [{"role": "user", "content": "ancient text"}],
                            day="2026-01-01"))
        store.record(_entry("new", [{"role": "user", "content": "recent text"}],
                            day="2026-08-20"))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        result = prune(conn, "2026-06-01")
        assert result["requests_deleted"] == 1
        assert conn.execute("SELECT COUNT(*) FROM prompt_requests").fetchone()[0] == 1
        # The old message had no other referent, so it is gone too.
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        # And it must be gone from the search index, not just the table.
        # Asserted against messages_fts directly as well as through search():
        # search() joins messages, so it would report [] even for a stale index
        # row -- and a stale index row is exactly what must not survive.
        assert conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ('"ancient"',)).fetchall() == []
        assert conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ('"recent"',)).fetchall() != []
        assert search(conn, "ancient") == []
        assert len(search(conn, "recent")) == 1
    finally:
        store.close()


def test_prune_keeps_a_message_still_referenced_by_a_newer_request(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        shared = [{"role": "system", "content": "shared system prompt"}]
        store.record(_entry("old", list(shared), day="2026-01-01"))
        store.record(_entry("new", list(shared), day="2026-08-20"))
        store.flush()
        conn = open_db(str(tmp_path / "p.db"))
        prune(conn, "2026-06-01")
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        assert len(search(conn, "shared")) == 1
    finally:
        store.close()


def test_recording_never_raises_on_bad_input(tmp_path):
    store = PromptStore(str(tmp_path / "p.db"))
    try:
        store.record({"request_id": "only-a-key"})   # no messages, no ts
        store.record({})                              # nothing at all
        store.flush()
        assert store.dropped >= 1
    finally:
        store.close()


def test_unwritable_path_degrades_silently(tmp_path):
    store = PromptStore(str(tmp_path / "nope" / "deeper" / "p.db"))
    try:
        store.record(_entry("r1", [{"role": "user", "content": "hi"}]))
        store.flush()
    finally:
        store.close()  # must not raise
