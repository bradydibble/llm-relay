"""Log buffer + streaming endpoints (plan 7): /logs and /logs/stream."""
from __future__ import annotations

import logging

import httpx

from llm_relay.api.app import create_app
from llm_relay.logbuffer import LogBuffer, install_log_buffer


def _min_cfg(tmp_path):
    (tmp_path / "providers.yaml").write_text("providers: {}\n")
    (tmp_path / "models.yaml").write_text("models: {}\n")
    return tmp_path


# --- LogBuffer unit --------------------------------------------------------

def test_logbuffer_recent_and_since():
    buf = LogBuffer(maxlen=5)
    for i in range(7):
        buf.append(f"line{i}")
    # maxlen drops the two oldest.
    assert buf.recent(limit=3) == ["line4", "line5", "line6"]
    head = buf.head_seq
    buf.append("line7")
    assert [l for _, l in buf.since(head)] == ["line7"]


def test_install_log_buffer_captures_llm_relay_logs():
    buf = install_log_buffer(maxlen=50)
    logging.getLogger("llm_relay").warning("captured-marker")
    assert any("captured-marker" in line for line in buf.recent())


# --- endpoints -------------------------------------------------------------

async def test_logs_endpoint_returns_buffered_lines(tmp_path):
    app = create_app(config_dir=_min_cfg(tmp_path))
    logging.getLogger("llm_relay").warning("hello-logs-endpoint")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/logs")
    assert resp.status_code == 200
    assert "hello-logs-endpoint" in resp.text


def test_logs_routes_registered(tmp_path):
    # The SSE generator reuses the proven StreamingResponse path; an in-process
    # infinite-stream read deadlocks on close under ASGITransport, so we assert
    # the routes are wired and rely on the buffer unit tests for the line logic.
    app = create_app(config_dir=_min_cfg(tmp_path))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/logs" in paths
    assert "/logs/stream" in paths
