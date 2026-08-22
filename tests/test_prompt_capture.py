"""Prompt content is captured only when explicitly enabled, and it shares a
request_id with the usage row so the two can be joined."""
from __future__ import annotations

import pytest


@pytest.fixture()
def dbs(tmp_path, monkeypatch):
    from llm_relay import prompt_store, usage_store

    usage_db = str(tmp_path / "usage.db")
    prompt_db = str(tmp_path / "prompts.db")
    monkeypatch.setenv("LLM_RELAY_USAGE_DB", usage_db)
    monkeypatch.setenv("LLM_RELAY_PROMPT_DB", prompt_db)
    usage_store.reset_store_for_tests()
    prompt_store.reset_store_for_tests()
    yield usage_db, prompt_db
    usage_store.reset_store_for_tests()
    prompt_store.reset_store_for_tests()


def _emit(**over):
    from llm_relay.api.instrumentation import emit_chat_completion

    kwargs = dict(
        request_body={"messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "what is warewulf"},
        ], "model": "main"},
        response_body={"usage": {"prompt_tokens": 40, "completion_tokens": 8}},
        response_text="a provisioning tool", usage=None,
        model_resolved="glm-5.2", provider_name="gb200", user_agent="pytest",
        start_ns=1_000_000_000, end_ns=2_000_000_000, status_code=200,
        streamed=False, outcome="success", client="claude-code",
        principal="brady",
    )
    kwargs.update(over)
    emit_chat_completion(**kwargs)


def test_messages_are_captured_when_enabled(dbs):
    from llm_relay import prompt_store

    _emit()
    prompt_store.get_store().flush()
    conn = prompt_store.open_db(dbs[1])
    n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 2


def test_usage_and_prompt_rows_share_a_request_id(dbs):
    from llm_relay import prompt_store, usage_store

    _emit()
    usage_store.get_store().flush()
    prompt_store.get_store().flush()
    uconn = usage_store.open_db(dbs[0])
    pconn = prompt_store.open_db(dbs[1])
    usage_id = uconn.execute("SELECT request_id FROM requests").fetchone()[0]
    prompt_id = pconn.execute("SELECT request_id FROM prompt_requests").fetchone()[0]
    assert usage_id == prompt_id


def test_completion_and_reasoning_text_are_captured(dbs):
    from llm_relay import prompt_store

    _emit(usage={"prompt_tokens": 10, "completion_tokens": 20,
                 "_reasoning_content": "let me think"},
          response_body=None, streamed=True, response_text="the answer")
    prompt_store.get_store().flush()
    conn = prompt_store.open_db(dbs[1])
    got = prompt_store.read_request(
        conn, conn.execute("SELECT request_id FROM prompt_requests").fetchone()[0])
    assert got["completion"] == "the answer"
    assert got["reasoning"] == "let me think"


def test_non_streamed_completion_comes_from_the_response_body(dbs):
    # The non-streaming path passes response_text=None and puts the answer in
    # response_body (app.py:1585-1589). Reading response_text alone would
    # archive every non-streamed request as a prompt with no answer.
    from llm_relay import prompt_store

    _emit(response_text=None, streamed=False, usage=None, response_body={
        "choices": [{"message": {"role": "assistant",
                                 "content": "warewulf provisions nodes",
                                 "reasoning_content": "recalling the docs"}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 8},
    })
    prompt_store.get_store().flush()
    conn = prompt_store.open_db(dbs[1])
    got = prompt_store.read_request(
        conn, conn.execute("SELECT request_id FROM prompt_requests").fetchone()[0])
    assert got["completion"] == "warewulf provisions nodes"
    assert got["reasoning"] == "recalling the docs"


def test_multimodal_content_parts_do_not_archive_the_payload(dbs):
    # An image part must reduce to a marker, not land in the archive as base64.
    from llm_relay import prompt_store

    blob = "A" * 512
    _emit(request_body={"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is in this picture"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + blob}},
    ]}], "model": "main"})
    prompt_store.get_store().flush()
    conn = prompt_store.open_db(dbs[1])
    got = prompt_store.read_request(
        conn, conn.execute("SELECT request_id FROM prompt_requests").fetchone()[0])
    content = got["messages"][0]["content"]
    assert "what is in this picture" in content
    assert blob not in content


def test_no_capture_when_prompt_db_is_unset(tmp_path, monkeypatch):
    from llm_relay import prompt_store, usage_store

    monkeypatch.setenv("LLM_RELAY_USAGE_DB", str(tmp_path / "u.db"))
    monkeypatch.delenv("LLM_RELAY_PROMPT_DB", raising=False)
    usage_store.reset_store_for_tests()
    prompt_store.reset_store_for_tests()
    try:
        _emit()  # must not raise
        assert prompt_store.get_store() is None
        assert not (tmp_path / "prompts.db").exists()
    finally:
        usage_store.reset_store_for_tests()
        prompt_store.reset_store_for_tests()


def test_capture_failure_never_breaks_the_request(dbs, monkeypatch):
    from llm_relay import prompt_store

    def _boom(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(prompt_store.PromptStore, "record", _boom)
    _emit()  # must not raise
