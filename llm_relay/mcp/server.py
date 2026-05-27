"""FastMCP server — exposes llm-relay status and model info as MCP tools.

Mounted at /mcp inside the main FastAPI app.  Clients connect with the
Streamable HTTP transport:

    http://<host>:<port>/mcp/mcp   (POST / SSE endpoint)

Configured tool list:
  relay_status    — active mode, alias resolutions, backend health
  list_models     — all models with current availability
  describe_alias  — resolve a single alias: current model, context_window,
                    members, and saturation flag

Important: the MCP session manager must be started in the parent app's
lifespan.  ``build_mcp_server()`` returns the FastMCP instance and the
session manager so the caller can do::

    mcp_instance, session_mgr = build_mcp_server(base_url)
    starlette_app = mcp_instance.streamable_http_app()
    # ... in the FastAPI lifespan:
    async with session_mgr.run():
        yield
"""
from __future__ import annotations

from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

_BASE_URL: str = "http://127.0.0.1:8090"
_mcp_instance: "FastMCP | None" = None  # set by build_mcp_server; used by tests


async def _get(path: str) -> Any:
    """Fetch *path* from the relay and return the parsed JSON body.

    Module-level so that tests can monkeypatch ``llm_relay.mcp.server._get``
    without needing to intercept the HTTP layer.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{_BASE_URL}{path}")
        r.raise_for_status()
        return r.json()


def build_mcp_server(
    base_url: str = _BASE_URL,
) -> tuple[Any, StreamableHTTPSessionManager]:
    """Build the FastMCP server.  Returns ``(starlette_app, session_manager)``.

    The *starlette_app* is the ASGI sub-app to mount (e.g. ``app.mount("/mcp", starlette_app)``).
    The *session_manager* must be started (via ``async with session_manager.run()``) in
    the parent FastAPI app's lifespan before any MCP requests are handled.
    """
    global _BASE_URL, _mcp_instance
    _BASE_URL = base_url

    mcp = FastMCP(
        name="llm-relay",
        instructions=(
            "Use relay_status to check the current LLM routing mode and which "
            "models are active before choosing a model for a task. "
            "Use list_models to enumerate all configured models with availability. "
            "Use describe_alias to inspect a specific alias before sending a request: "
            "it tells you the resolved model, context window, and whether the backend "
            "is saturated."
        ),
    )

    @mcp.tool()
    async def relay_status() -> dict[str, Any]:
        """Return the current llm-relay routing state.

        Includes:
        - mode: matched preset mode name(s) based on running backends, or
          ["custom"] when the active set doesn't match any defined preset
        - available_local_models: model ids currently reachable on the local
          inference host
        - aliases: each routing alias and the model it currently resolves to
        - backends: per-backend health status and model list
        """
        return await _get("/status")

    @mcp.tool()
    async def list_models() -> list[dict[str, Any]]:
        """Return all configured models with their current availability status.

        Each entry includes id, provider, class, status (available / unavailable /
        degraded), context_window, capabilities, tags, and privacy level.
        """
        data = await _get("/v1/available-models")
        models = []
        for name, info in data.items():
            if name in ("aliases", "alias_info"):
                continue
            models.append({"id": name, **info})
        models.sort(key=lambda m: (m.get("status") != "available", m["id"]))
        return models

    @mcp.tool()
    async def describe_alias(name: str) -> dict[str, Any]:
        """Return the current resolution for a routing alias.

        Includes the resolved concrete model, its context_window, the full
        ordered member list, and a saturated flag (true if the current backend
        has no in-flight slots free and 503 + Retry-After would be returned).

        Use this BEFORE constructing a /v1/chat/completions request so you can:
          - size your prompt to the actual context window of the live model
          - back off to a different alias if this one is saturated

        Args:
            name: alias name (e.g. 'main', 'fast', 'long-context').
        """
        payload = await _get("/v1/available-models")
        alias_info = payload.get("alias_info", {})

        if name not in alias_info:
            return {
                "alias": name,
                "error": f"unknown alias '{name}'",
                "available_aliases": list(payload.get("aliases", {}).keys()),
            }

        info = alias_info[name]
        current_model = info["current"]

        # Derive saturation from /status.  If per-backend inflight counters are
        # absent (Task 9-or-later enhancement), default to False — never crash.
        saturated = False
        status = await _get("/status")
        for _backend_key, backend in status.get("backends", {}).items():
            if current_model in backend.get("models", []):
                used = backend.get("inflight_used")
                capacity = backend.get("inflight_capacity")
                if used is not None and capacity is not None:
                    saturated = used >= capacity
                break

        return {
            "alias": name,
            "current": current_model,
            "context_window": info["context_window"],
            "members": info["members"],
            "saturated": saturated,
        }

    # streamable_http_app() lazily initialises the session_manager; call it
    # once here so the returned session_manager is non-None.  The caller uses
    # the returned starlette_app as the ASGI sub-app to mount.
    starlette_app = mcp.streamable_http_app()
    _mcp_instance = mcp  # expose for test introspection
    return starlette_app, mcp.session_manager


# Backwards-compat alias (returns app only; caller is responsible for the lifespan)
build_mcp_app = lambda base_url=_BASE_URL: build_mcp_server(base_url)[0]  # noqa: E731
