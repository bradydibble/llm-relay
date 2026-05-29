"""Verify stream_request has a bounded read timeout so stalled upstreams don't hang forever."""
from __future__ import annotations

import inspect

from llm_relay.routing import router as router_mod


def test_stream_request_read_timeout_is_bounded():
    """read=None caused unbounded hangs; we want a finite read timeout in the stream path."""
    src = inspect.getsource(router_mod.RequestRouter.stream_request)
    # Look for httpx.Timeout(... read=...). The value must not be None.
    assert "read=None" not in src, (
        "stream_request must not use read=None — that lets the relay hang on a stalled upstream "
        "indefinitely, which is the cascade trigger this bounded read timeout guards against"
    )
    # Sanity: still references httpx.Timeout in the stream path.
    assert "httpx.Timeout" in src
