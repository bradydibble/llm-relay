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


def test_forward_request_non_stream_timeout_is_generous():
    """The NON-streaming path (forward_request) capped the whole request at a 300s
    TOTAL timeout, which silently overrode a caller's longer client timeout and killed
    any completion that ran past 300s. On the local 35B a large (~70k) prompt prefills
    100-250s+ before generation even starts, so a 300s total is the cap that kills slow
    but valid work — exactly the arbitrary cutoff we do not want on idle hardware.

    It must use a structured timeout: a GENEROUS read window for slow large completions,
    but a SHORT connect so a genuinely dead backend still fails fast instead of holding
    a slot for the full window (the saturation failure mode).
    """
    src = inspect.getsource(router_mod.RequestRouter.forward_request)
    assert "timeout=300.0" not in src, (
        "forward_request must not cap the non-stream request at a tight 300s TOTAL — it "
        "overrode the engine's 900s client timeout and killed completions over 300s"
    )
    assert "httpx.Timeout" in src, "use a structured httpx.Timeout (separate read vs connect)"
    assert "read=900.0" in src, "give a generous 900s read window for slow large completions"
    assert "connect=10.0" in src, "keep connect short so a dead backend fails fast, not slot-held"
