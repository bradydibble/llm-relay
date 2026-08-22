"""Prune CLI and growth visibility for the prompt store.

Retention is indefinite by decision, so the prune path must exist and be
tested BEFORE it is ever needed -- and it must do nothing with no
configuration at all. ``LLM_RELAY_PROMPT_RETENTION_DAYS`` unset (or 0) means
keep forever; only an explicit ``--older-than`` or a positive retention window
deletes anything.

Every fixture here is built through ``PromptStore.record`` -- the real writer
-- rather than hand-inserted SQL. Each real bug in this store came from a test
fixture more generous than the production caller, so a test that cannot see the
shape production actually produces is not a test.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta

import pytest

from llm_relay import cli, metrics
from llm_relay.metrics import PromptStoreCollector
from llm_relay.prompt_store import PromptStore, open_db


def _entry(request_id, messages, *, day="2026-08-20", ts=1787000000.0,
           principal="brady", client="claude-code", model="glm-5.2",
           completion="an answer", reasoning=""):
    return {
        "request_id": request_id, "ts": ts, "day": day, "principal": principal,
        "client": client, "model": model, "messages": messages,
        "completion": completion, "reasoning": reasoning,
    }


def _store_with(tmp_path, entries) -> str:
    """Build a fixture store through the real writer, then close it."""
    path = str(tmp_path / "p.db")
    store = PromptStore(path)
    try:
        for entry in entries:
            store.record(entry)
        store.flush()
    finally:
        store.close()
    return path


def _run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["llm-relay", *argv])
    return cli.main()


def _norm(text: str) -> str:
    """Collapse rich's line wrapping so substring asserts are width-stable."""
    return " ".join(text.split())


def _rows(path, sql):
    conn = open_db(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _samples(collector):
    out = {}
    for family in collector.collect():
        for sample in family.samples:
            out[sample.name] = sample.value
    return out


# A shared system prompt plus a per-request question: the shape that makes
# content addressing matter, and the shape prune's orphan rule is about.
_SHARED = {"role": "system", "content": "you are the fleet assistant"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_RELAY_PROMPT_DB", raising=False)
    monkeypatch.delenv("LLM_RELAY_PROMPT_RETENTION_DAYS", raising=False)
    # Keep audit lines out of the repo's config/ directory.
    monkeypatch.setenv("LLM_RELAY_AUDIT_LOG", str(tmp_path / "audit.log"))


def _two_days(tmp_path):
    return _store_with(tmp_path, [
        _entry("old", [dict(_SHARED),
                       {"role": "user", "content": "ancient warewulf question"}],
               day="2026-01-05"),
        _entry("new", [dict(_SHARED),
                       {"role": "user", "content": "todays substrate question"}],
               day="2026-08-20"),
    ])


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #

def test_prune_deletes_old_requests_and_reports_counts(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)

    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-08-01")
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "deleted 1 request" in out
    assert "1 stored message" in out
    assert [r[0] for r in _rows(path, "SELECT request_id FROM prompt_requests")] == ["new"]
    # The shared system prompt survives: a newer request still references it.
    kept = {r[0] for r in _rows(path, "SELECT content FROM messages")}
    assert len(kept) == 2


def test_prune_writes_an_audit_event(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)
    log = os.environ["LLM_RELAY_AUDIT_LOG"]

    _run(monkeypatch, "prompts", "prune", "--db", path, "--older-than", "2026-08-01")
    capsys.readouterr()

    with open(log) as fh:
        body = fh.read()
    assert "prompt_prune" in body
    assert "2026-08-01" in body


def test_dry_run_reports_the_counts_a_real_prune_then_delivers(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)

    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-08-01", "--dry-run")
    preview = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "would delete 1 request(s) and 1 stored message(s)" in preview
    # Nothing written.
    assert len(_rows(path, "SELECT request_id FROM prompt_requests")) == 2
    assert _rows(path, "SELECT COUNT(*) FROM messages")[0][0] == 3

    # The preview is only worth printing if the real run agrees with it.
    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-08-01")
    real = _norm(capsys.readouterr().out)
    assert rc == 0
    assert "deleted 1 request(s) and 1 stored message(s)" in real


def test_dry_run_preview_counts_a_shared_message_only_when_nothing_survives(
        monkeypatch, capsys, tmp_path):
    # Both requests age out together, so the shared system prompt is orphaned
    # too: 2 requests, 3 stored messages, all of them gone.
    path = _store_with(tmp_path, [
        _entry("a", [dict(_SHARED), {"role": "user", "content": "first question"}],
               day="2026-01-05"),
        _entry("b", [dict(_SHARED), {"role": "user", "content": "second question"}],
               day="2026-01-06"),
    ])

    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-08-01", "--dry-run")
    preview = _norm(capsys.readouterr().out)
    assert rc == 0
    assert "would delete 2 request(s) and 3 stored message(s)" in preview

    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-08-01")
    real = _norm(capsys.readouterr().out)
    assert rc == 0
    assert "deleted 2 request(s) and 3 stored message(s)" in real
    assert _rows(path, "SELECT COUNT(*) FROM messages")[0][0] == 0


def test_invalid_date_exits_non_zero_and_deletes_nothing(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)

    rc = _run(monkeypatch, "prompts", "prune", "--db", path,
              "--older-than", "2026-13-45")
    out = _norm(capsys.readouterr().out)

    assert rc == 1
    assert "YYYY-MM-DD" in out
    assert len(_rows(path, "SELECT request_id FROM prompt_requests")) == 2


def test_prune_does_nothing_when_retention_is_unset(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)

    rc = _run(monkeypatch, "prompts", "prune", "--db", path)
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "indefinite" in out
    assert len(_rows(path, "SELECT request_id FROM prompt_requests")) == 2
    assert _rows(path, "SELECT COUNT(*) FROM messages")[0][0] == 3


def test_retention_zero_also_means_keep_forever(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)
    monkeypatch.setenv("LLM_RELAY_PROMPT_RETENTION_DAYS", "0")

    rc = _run(monkeypatch, "prompts", "prune", "--db", path)
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "indefinite" in out
    assert len(_rows(path, "SELECT request_id FROM prompt_requests")) == 2


def test_a_positive_retention_window_sets_the_cutoff(monkeypatch, capsys, tmp_path):
    today = date.today()
    path = _store_with(tmp_path, [
        _entry("stale", [{"role": "user", "content": "last years question"}],
               day=(today - timedelta(days=400)).isoformat()),
        _entry("fresh", [{"role": "user", "content": "this weeks question"}],
               day=today.isoformat()),
    ])
    monkeypatch.setenv("LLM_RELAY_PROMPT_RETENTION_DAYS", "30")

    rc = _run(monkeypatch, "prompts", "prune", "--db", path)
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "deleted 1 request" in out
    assert [r[0] for r in _rows(path, "SELECT request_id FROM prompt_requests")] == ["fresh"]


def test_unparseable_retention_value_exits_non_zero(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)
    monkeypatch.setenv("LLM_RELAY_PROMPT_RETENTION_DAYS", "banana")

    rc = _run(monkeypatch, "prompts", "prune", "--db", path)
    out = _norm(capsys.readouterr().out)

    assert rc == 1
    assert "LLM_RELAY_PROMPT_RETENTION_DAYS" in out
    assert len(_rows(path, "SELECT request_id FROM prompt_requests")) == 2


def test_prune_on_a_store_that_does_not_exist_creates_nothing(monkeypatch, capsys, tmp_path):
    ghost = tmp_path / "never.db"

    rc = _run(monkeypatch, "prompts", "prune", "--db", str(ghost),
              "--older-than", "2026-08-01")
    capsys.readouterr()

    assert rc == 0
    assert not ghost.exists()


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def test_stats_prints_the_dedup_ratio_and_byte_size(monkeypatch, capsys, tmp_path):
    history = [
        {"role": "system", "content": "you are the fleet assistant"},
        {"role": "user", "content": "what is a substrate"},
    ]
    # Three turns resending the same two messages: 6 links over 2 stored
    # messages, so the dedup ratio -- the prompt-cache opportunity -- is 3.
    path = _store_with(tmp_path, [
        _entry("r1", [dict(m) for m in history]),
        _entry("r2", [dict(m) for m in history]),
        _entry("r3", [dict(m) for m in history]),
    ])

    rc = _run(monkeypatch, "prompts", "stats", "--db", path)
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "3.00x" in out
    assert "prompt-cache opportunity" in out
    assert "On disk" in out
    match = re.search(r"([\d,]+) bytes", out)
    assert match is not None
    assert int(match.group(1).replace(",", "")) > 0


def test_stats_reads_the_db_path_from_the_environment(monkeypatch, capsys, tmp_path):
    path = _two_days(tmp_path)
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", path)

    rc = _run(monkeypatch, "prompts", "stats")
    out = _norm(capsys.readouterr().out)

    assert rc == 0
    assert "Requests" in out


def test_stats_without_a_configured_db_exits_non_zero(monkeypatch, capsys):
    rc = _run(monkeypatch, "prompts", "stats")
    out = _norm(capsys.readouterr().out)

    assert rc == 1
    assert "LLM_RELAY_PROMPT_DB" in out


# --------------------------------------------------------------------------- #
# growth gauges
# --------------------------------------------------------------------------- #

def test_gauges_report_store_bytes_and_stored_message_count(monkeypatch, tmp_path):
    history = [
        {"role": "system", "content": "you are the fleet assistant"},
        {"role": "user", "content": "what is a substrate"},
    ]
    path = _store_with(tmp_path, [
        _entry("r1", [dict(m) for m in history]),
        _entry("r2", [dict(m) for m in history]),
    ])
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", path)

    got = _samples(PromptStoreCollector())

    # Content-addressed: two resends of the same conversation are still two
    # stored messages, which is exactly what makes the gauge meaningful.
    assert got["llm_relay_prompt_store_messages"] == 2.0
    assert got["llm_relay_prompt_store_bytes"] > 0


def test_gauges_are_silent_and_create_nothing_when_capture_is_off(monkeypatch, tmp_path):
    families = list(PromptStoreCollector().collect())
    assert {f.name for f in families} == {
        "llm_relay_prompt_store_bytes", "llm_relay_prompt_store_messages"}
    assert all(not f.samples for f in families)

    # A configured-but-absent database must not be created by a scrape.
    ghost = tmp_path / "never.db"
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", str(ghost))
    assert all(not f.samples for f in PromptStoreCollector().collect())
    assert not ghost.exists()


def test_gauges_are_registered_on_the_relay_registry(monkeypatch, tmp_path):
    path = _store_with(tmp_path, [
        _entry("r1", [{"role": "user", "content": "one question"}]),
    ])
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", path)

    body, _ = metrics.render_exposition()

    assert b"llm_relay_prompt_store_bytes" in body
    assert b"llm_relay_prompt_store_messages 1.0" in body
