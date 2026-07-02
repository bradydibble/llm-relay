"""MCP is served at exactly /mcp (no doubled /mcp/mcp)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from llm_relay.api.app import create_app

_HDRS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"},
    },
}


def test_mcp_served_at_clean_path(tmp_path):
    app = create_app(config_dir=tmp_path)
    with TestClient(app) as c:  # context manager runs lifespan (session manager)
        r = c.post("/mcp", json=_INIT, headers=_HDRS)
        assert r.status_code != 404
        old = c.post("/mcp/mcp", json=_INIT, headers=_HDRS)
        assert old.status_code in (404, 307)
