"""FastMCP server — exposes llm-relay status and model info as MCP tools.

Mounted at /mcp inside the main FastAPI app.  Clients connect with the
Streamable HTTP transport:

    http://<host>:<port>/mcp/mcp   (POST / SSE endpoint)

Configured tool list:
  relay_status   — active mode, alias resolutions, backend health
  list_models    — all models with current availability

Important: the MCP session manager must be started in the parent app's
lifespan.  ``build_mcp_server()`` returns both the ASGI sub-app and the
session manager so the caller can do::

    mcp_starlette, session_mgr = build_mcp_server(base_url)
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


def build_mcp_server(
    base_url: str = _BASE_URL,
) -> tuple[Any, StreamableHTTPSessionManager]:
    """Build the FastMCP server.  Returns ``(starlette_app, session_manager)``.

    The *session_manager* must be started (via ``async with session_manager.run()``) in
    the parent FastAPI app's lifespan before any MCP requests are handled.
    """
    mcp = FastMCP(
        name="llm-relay",
        instructions=(
            "Use relay_status to check the current LLM routing mode and which "
            "models are active before choosing a model for a task. "
            "Use list_models to enumerate all configured models with availability."
        ),
    )

    async def _get(path: str) -> Any:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url}{path}")
            r.raise_for_status()
            return r.json()

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

    starlette_app = mcp.streamable_http_app()
    return starlette_app, mcp.session_manager


# Backwards-compat alias (returns app only; caller is responsible for the lifespan)
build_mcp_app = lambda base_url=_BASE_URL: build_mcp_server(base_url)[0]  # noqa: E731
