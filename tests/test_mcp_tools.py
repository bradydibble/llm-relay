"""Verify MCP tool shapes and behavior."""
from __future__ import annotations

from typing import Any

import pytest

# The MCP SDK is an optional extra (install with `.[mcp]`). When it isn't
# present, skip this whole module instead of erroring — the relay degrades
# the same way (app.py guards `from ..mcp import build_mcp_server`).
pytest.importorskip("mcp")


def _stub_payloads() -> dict[str, Any]:
    """Two-route response: '/v1/available-models' and '/status'."""
    return {
        "/v1/available-models": {
            "qwen3.6-35b-a3b": {
                "provider": "local-llm",
                "status": "available",
                "context_window": 131072,
                "capabilities": ["tool_use", "structured_output"],
                "tags": ["local"],
                "privacy": "local_only",
                "preference": 1.0,
            },
            "qwen3.5-9b": {
                "provider": "local-llm",
                "status": "available",
                "context_window": 65536,
                "capabilities": ["tool_use"],
                "tags": ["local"],
                "privacy": "local_only",
                "preference": 0.5,
            },
            "cloud-test": {
                "provider": "cloud-provider",
                "status": "available",
                "context_window": 200000,
                "capabilities": ["tool_use", "structured_output"],
                "tags": ["cloud"],
                "privacy": "cloud_ok",
                "preference": 0.8,
            },
            "aliases": {"main": ["qwen3.6-35b-a3b", "qwen3.5-9b"]},
            "alias_info": {
                "main": {
                    "members": ["qwen3.6-35b-a3b", "qwen3.5-9b"],
                    "current": "qwen3.6-35b-a3b",
                    "context_window": 131072,
                }
            },
        },
        "/status": {
            "mode": ["test"],
            "available_local_models": ["qwen3.6-35b-a3b", "qwen3.5-9b"],
            "aliases": {"main": "qwen3.6-35b-a3b"},
            "backends": {
                "local-llm:8081": {
                    "status": "healthy",
                    "models": ["qwen3.6-35b-a3b"],
                    # no inflight_used/inflight_capacity — verify graceful fallback
                },
                "local-llm:8080": {
                    "status": "healthy",
                    "models": ["qwen3.5-9b"],
                },
            },
        },
    }


def _get_tool(mcp_server, name: str):
    """Extract a registered MCP tool's underlying async function.

    FastMCP version surface varies; try a couple of access paths.
    """
    if hasattr(mcp_server, "_tool_manager") and hasattr(mcp_server._tool_manager, "_tools"):
        entry = mcp_server._tool_manager._tools[name]
        return entry.fn if hasattr(entry, "fn") else entry
    # Fall back: directly iterate registered tools
    raise AssertionError(f"could not find MCP tool {name}")


async def test_describe_alias_returns_resolution_for_known_alias(monkeypatch):
    """describe_alias('main') → {alias, current, context_window, members, saturated}."""
    from llm_relay.mcp import server as mcp_mod

    payloads = _stub_payloads()

    async def fake_get(path: str):
        return payloads[path]

    monkeypatch.setattr(mcp_mod, "_get", fake_get, raising=False)

    _starlette_app, _mgr = mcp_mod.build_mcp_server(base_url="http://test")
    fn = _get_tool(mcp_mod._mcp_instance, "describe_alias")

    result = await fn(name="main")
    assert result["alias"] == "main"
    assert result["current"] == "qwen3.6-35b-a3b"
    assert result["context_window"] == 131072
    assert result["members"] == ["qwen3.6-35b-a3b", "qwen3.5-9b"]
    assert result["saturated"] is False  # status payload had no inflight fields


async def test_describe_alias_unknown_alias_returns_error_and_lists_available(monkeypatch):
    """Unknown alias returns {alias, error, available_aliases}."""
    from llm_relay.mcp import server as mcp_mod

    payloads = _stub_payloads()

    async def fake_get(path: str):
        return payloads[path]

    monkeypatch.setattr(mcp_mod, "_get", fake_get, raising=False)

    _starlette_app, _mgr = mcp_mod.build_mcp_server(base_url="http://test")
    fn = _get_tool(mcp_mod._mcp_instance, "describe_alias")

    result = await fn(name="nonexistent-alias")
    assert result["alias"] == "nonexistent-alias"
    assert "error" in result and "nonexistent-alias" in result["error"]
    assert "main" in result["available_aliases"]


async def test_select_for_capability_filters_by_min_context_window(monkeypatch):
    """min_context_window=100000 excludes qwen3.5-9b (64K) but keeps qwen3.6-35b-a3b (128K)."""
    from llm_relay.mcp import server as mcp_mod

    payloads = _stub_payloads()

    async def fake_get(path: str):
        return payloads[path]

    monkeypatch.setattr(mcp_mod, "_get", fake_get, raising=False)

    _starlette_app, _mgr = mcp_mod.build_mcp_server(base_url="http://test")
    fn = _get_tool(mcp_mod._mcp_instance, "select_for_capability")

    result = await fn(min_context_window=100000, privacy="local_only")
    assert "qwen3.5-9b" not in result["candidates"]
    assert "qwen3.6-35b-a3b" in result["candidates"]
    assert result["best"] == "qwen3.6-35b-a3b"
    assert "100000" in result["rationale"]


async def test_select_for_capability_excludes_cloud_when_local_only(monkeypatch):
    """privacy='local_only' filters out cloud models even if they'd otherwise qualify."""
    from llm_relay.mcp import server as mcp_mod

    payloads = _stub_payloads()

    async def fake_get(path: str):
        return payloads[path]

    monkeypatch.setattr(mcp_mod, "_get", fake_get, raising=False)

    _starlette_app, _mgr = mcp_mod.build_mcp_server(base_url="http://test")
    fn = _get_tool(mcp_mod._mcp_instance, "select_for_capability")

    # cloud-test has 200K context and tool_use, but privacy=cloud_ok — must be excluded
    result = await fn(
        min_context_window=0,
        requires_capabilities=["tool_use"],
        privacy="local_only",
    )
    assert "cloud-test" not in result["candidates"]
    # both local models have tool_use and should be present
    assert "qwen3.6-35b-a3b" in result["candidates"]
    assert "qwen3.5-9b" in result["candidates"]
    # 35b has preference=1.0, 9b has preference=0.5 → 35b should be best
    assert result["best"] == "qwen3.6-35b-a3b"


async def test_select_for_capability_returns_empty_when_no_match(monkeypatch):
    """When no model satisfies constraints, returns {candidates: [], best: None}."""
    from llm_relay.mcp import server as mcp_mod

    payloads = _stub_payloads()

    async def fake_get(path: str):
        return payloads[path]

    monkeypatch.setattr(mcp_mod, "_get", fake_get, raising=False)

    _starlette_app, _mgr = mcp_mod.build_mcp_server(base_url="http://test")
    fn = _get_tool(mcp_mod._mcp_instance, "select_for_capability")

    # Impossibly large context window — no model satisfies this
    result = await fn(min_context_window=10_000_000, privacy="cloud_ok")
    assert result["candidates"] == []
    assert result["best"] is None
    assert "rationale" in result
