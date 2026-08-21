"""FastMCP server — exposes llm-relay status and model info as MCP tools.

Mounted at /mcp inside the main FastAPI app.  Clients connect with the
Streamable HTTP transport:

    http://<host>:<port>/mcp/   (POST / SSE endpoint; exact /mcp 307-redirects
                                 here, and a reverse proxy can normalize it)

Configured tool list:
  relay_status           — active mode, alias resolutions, backend health
  list_models            — all models with current availability
  describe_alias         — resolve a single alias: current model, context_window,
                           members, and saturation flag
  select_for_capability  — find models matching constraints (context, capabilities,
                           privacy), ranked by preference

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
        # Serve at the sub-app root so the FastAPI mount at /mcp yields a clean
        # /mcp endpoint (not the SDK-default doubled /mcp/mcp).
        streamable_http_path="/",
        instructions=(
            "Use relay_status to check the current LLM routing mode and which "
            "models are active before choosing a model for a task. "
            "Use list_models to enumerate all configured models with availability. "
            "Use describe_alias to inspect a specific alias before sending a request: "
            "it tells you the resolved model, context window, and whether the backend "
            "is saturated. "
            "Use select_for_capability to find models that meet your requirements "
            "(minimum context window, required capability tags, privacy constraints) "
            "without enumerating models yourself — it returns ranked candidates so "
            "you can pick the best fit and then request it by name."
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
        """Return all models and aliases available on the relay RIGHT NOW, with
        the host-qualified IDs and rendered capabilities a client needs to
        select a model for /v1/chat/completions.

        Each entry includes:
        - id: the model/alias ID to use in the 'model' field of a request
          (host-qualified for concrete models: 'llama-01:ornith-35b',
          'amd-mi300x:qwen3.6-35b'; bare name for aliases: 'main', 'fast')
        - capabilities: rendered booleans {toolcall, reasoning, structured_output}
          — an alias advertises the INTERSECTION of its members' capabilities
        - context_length / max_model_len: the live context window
        - owned_by: 'llm-relay-alias' for aliases, provider name for concrete models

        Use the 'id' field directly as the 'model' parameter in your request.
        """
        data = await _get("/v1/models")
        models = data.get("data", [])
        # Sort: available first, then by id
        models.sort(key=lambda m: m.get("id", ""))
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

    @mcp.tool()
    async def select_for_capability(
        min_context_window: int = 0,
        requires_capabilities: list[str] | None = None,
        privacy: str = "local_only",
    ) -> dict[str, Any]:
        """Find models matching given constraints, ranked by preference.

        Uses the same /v1/models endpoint that OpenAI clients use, so the
        returned IDs are directly usable as the 'model' field in a request.

        Args:
            min_context_window: only return models whose context_window >= this.
            requires_capabilities: list of capability names the model must support.
                These match the RENDERED capability booleans from /v1/models:
                'toolcall', 'reasoning', 'structured_output' (not the raw config
                tags like 'tool_use'). Default: no requirement.
            privacy: 'local_only' or 'cloud_ok'. Default local_only.
                Note: /v1/models does not expose privacy; this filter is a
                best-effort check against /v1/available-models. When unavailable,
                all models are returned regardless of privacy.

        Returns:
            {"candidates": [id, ...], "best": id_or_None, "rationale": "..."}
            The IDs are host-qualified (e.g. 'llama-01:ornith-35b') or alias
            names (e.g. 'main') — directly usable in a request.
        """
        data = await _get("/v1/models")
        models = data.get("data", [])

        # Privacy filter: cross-reference with /v1/available-models for the
        # privacy field (not exposed on /v1/models).
        privacy_map: dict[str, str] = {}
        try:
            avail = await _get("/v1/available-models")
            for name, info in avail.items():
                if isinstance(info, dict) and name not in ("aliases", "alias_info"):
                    privacy_map[name] = info.get("privacy", "")
        except Exception:
            pass  # /v1/available-models may be unavailable; skip privacy filter

        required_caps = set(requires_capabilities or [])
        candidates = []

        for m in models:
            model_id = m.get("id", "")
            caps = m.get("capabilities") or {}
            # caps is a dict of booleans: {toolcall: True, reasoning: True, ...}
            # Check that every required cap is present and True
            if required_caps:
                if not all(caps.get(c) for c in required_caps):
                    continue

            # Context window check
            cw = m.get("context_length") or m.get("max_model_len") or 0
            if cw < min_context_window:
                continue

            # Privacy filter (best-effort, against the bare name)
            if privacy == "local_only":
                bare_name = model_id.split(":")[-1] if ":" in model_id else model_id
                model_privacy = privacy_map.get(bare_name, "")
                if model_privacy and model_privacy != "local_only":
                    continue

            candidates.append(m)

        # Sort by id for stability
        candidates.sort(key=lambda m: m.get("id", ""))

        candidate_ids = [m.get("id", "") for m in candidates]
        return {
            "candidates": candidate_ids,
            "best": candidate_ids[0] if candidate_ids else None,
            "rationale": (
                f"filtered {len(candidate_ids)} of {len(models)} models by "
                f"min_context={min_context_window}, "
                f"requires={list(requires_capabilities or [])}, "
                f"privacy={privacy}"
            ),
        }

    # streamable_http_app() lazily initialises the session_manager; call it
    # once here so the returned session_manager is non-None.  The caller uses
    # the returned starlette_app as the ASGI sub-app to mount.
    starlette_app = mcp.streamable_http_app()
    _mcp_instance = mcp  # expose for test introspection
    return starlette_app, mcp.session_manager


# Backwards-compat alias (returns app only; caller is responsible for the lifespan)
build_mcp_app = lambda base_url=_BASE_URL: build_mcp_server(base_url)[0]  # noqa: E731
