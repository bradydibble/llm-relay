"""FastAPI application for llm-relay."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from ..config.loader import ConfigLoader
from ..config.types import SaturationError
from ..discovery.manager import DiscoveryManager
from ..routing.keys import compose_backend_key
from ..routing.router import RequestRouter
from .instrumentation import emit_chat_completion, reassemble_sse
from ..metrics import did_fall_back, metrics_enabled, register_discovery_collector, render_exposition, resolve_client, set_known_routable


def _resolve_base_url() -> str:
    """Externally-reachable root URL for A2A agent-card and MCP config."""
    return os.environ.get("LLM_RELAY_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    if config_dir:
        return Path(config_dir)
    env = os.environ.get("LLM_RELAY_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def _build_available_payload(cfg: ConfigLoader, disc: DiscoveryManager) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, m in cfg.models.models.items():
        status = disc.get_model_state(name)
        # Prefer live max_model_len from the backend (authoritative); fall back
        # to static models.yaml value when the backend isn't currently
        # reporting one (down, no max_model_len field, etc.).
        live_cw = disc.get_live_context_window(name)
        out[name] = {
            "provider": m.provider,
            "class": m.class_name,
            "status": status.value,
            "context_window": live_cw if live_cw is not None else m.context_window,
            "context_window_source": "live" if live_cw is not None else "config",
            "capabilities": m.capabilities,
            "tags": m.tags,
            "privacy": m.privacy.value,
            "port": m.port,
            "path": m.path,
        }
    out["aliases"] = dict(cfg.models.aliases)
    # Enriched per-alias metadata so clients can show context_window etc. for
    # aliases (which are otherwise just names). `current` is a display
    # approximation: the first member that discovery reports as available /
    # degraded, falling back to the first declared member. The selector applies
    # additional filters (privacy, min_context, require_tools) at request time,
    # so the actually-routed model may differ.
    alias_info: dict[str, Any] = {}
    for alias, members in cfg.models.aliases.items():
        members_list = list(members)
        current: str | None = None
        for member in members_list:
            if member not in cfg.models.models:
                continue
            if disc.get_model_state(member).value in ("available", "degraded"):
                current = member
                break
        if current is None:
            for member in members_list:
                if member in cfg.models.models:
                    current = member
                    break
        if current is not None:
            live_cw = disc.get_live_context_window(current)
            cw = live_cw if live_cw is not None else cfg.models.models[current].context_window
        else:
            cw = None
        alias_info[alias] = {
            "members": members_list,
            "current": current,
            "context_window": cw,
        }
    out["alias_info"] = alias_info
    return out


def create_app(config_dir: str | Path | None = None) -> FastAPI:
    cfg_path = _resolve_config_dir(config_dir)
    config = ConfigLoader(config_dir=cfg_path)
    config.load()
    discovery = DiscoveryManager()
    router = RequestRouter(config, discovery)

    from contextlib import AsyncExitStack

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            # Start the MCP session manager (no-op if MCP not installed)
            if _mcp_session_mgr is not None:
                await stack.enter_async_context(_mcp_session_mgr.run())
            # Register one polling client per (provider, port, path) combo.
            for provider_name, provider in config.providers.items():
                if not provider.enabled:
                    continue
                models_for_provider = {
                    name: m for name, m in config.models.models.items() if m.provider == provider_name
                }
                if not models_for_provider:
                    await discovery.register_backend(
                        key=provider_name,
                        provider_name=provider_name,
                        base_url=provider.base_url.rstrip("/"),
                        models_hint=[],
                        health_endpoint=provider.health_endpoint,
                        poll_interval=provider.poll_interval,
                        circuit_breaker=provider.circuit_breaker,
                        timeout=provider.health_check_timeout,
                        max_concurrent=provider.max_concurrent,
                    )
                    continue
                groups: dict[tuple[int | None, str], list[str]] = {}
                for name, m in models_for_provider.items():
                    groups.setdefault((m.port, m.path or ""), []).append(name)
                for (port, path), names in groups.items():
                    base = provider.base_url.rstrip("/")
                    if port:
                        base = f"{base}:{port}"
                    if path:
                        base = f"{base}/{path.lstrip('/')}"
                    key = compose_backend_key(provider_name, port, path)
                    await discovery.register_backend(
                        key=key,
                        provider_name=provider_name,
                        base_url=base,
                        models_hint=names,
                        health_endpoint=provider.health_endpoint,
                        poll_interval=provider.poll_interval,
                        circuit_breaker=provider.circuit_breaker,
                        timeout=provider.health_check_timeout,
                        max_concurrent=provider.max_concurrent,
                    )
            yield
        await discovery.shutdown()

    # --- MCP sub-app (optional dep) -----------------------------------
    _mcp_app = None
    _mcp_session_mgr = None
    try:
        from ..mcp import build_mcp_server
        _mcp_app, _mcp_session_mgr = build_mcp_server(base_url=_resolve_base_url())
    except ImportError:
        pass

    # --- A2A routes (optional dep) -------------------------------------
    _a2a_card_routes: list = []
    _a2a_jsonrpc_routes: list = []
    _a2a_rest_routes: list = []
    try:
        from ..a2a import build_a2a_routes
        _a2a_card_routes, _a2a_jsonrpc_routes, _a2a_rest_routes = build_a2a_routes(
            base_url=_resolve_base_url(),
        )
    except ImportError:
        pass

    app = FastAPI(title="llm-relay", version="1.0.0", lifespan=lifespan)
    app.state.config = config
    app.state.discovery = discovery
    app.state.router = router

    async def _available(request: Request) -> dict[str, Any]:
        return _build_available_payload(
            request.app.state.config, request.app.state.discovery
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        disc = request.app.state.discovery
        return {
            "status": "ok",
            "endpoints": {
                key: {
                    "status": c.state.status.value,
                    "last_poll": c.state.last_poll,
                    "models": c.state.models,
                }
                for key, c in disc.clients.items()
            },
        }

    @app.get("/available-models")
    async def available_models(request: Request) -> dict[str, Any]:
        return await _available(request)

    @app.get("/v1/available-models")
    async def available_models_v1(request: Request) -> dict[str, Any]:
        return await _available(request)

    @app.get("/v1/models")
    async def list_models_openai(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        data: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, m in cfg.models.models.items():
            if name in seen:
                continue
            seen.add(name)
            data.append({"id": name, "object": "model", "owned_by": m.provider})
        for alias in cfg.models.aliases.keys():
            if alias in seen:
                continue
            seen.add(alias)
            data.append({"id": alias, "object": "model", "owned_by": "llm-relay-alias"})
        return {"object": "list", "data": data}

    @app.get("/status")
    async def relay_status(request: Request) -> dict[str, Any]:
        cfg = request.app.state.config
        disc = request.app.state.discovery

        def _actually_available(model_name: str) -> bool:
            """True only if the model's backend is healthy AND reports a matching
            model id.  For models that share a port (port-mutex pairs), this
            distinguishes which one is actually loaded."""
            key = disc.model_to_client.get(model_name)
            if not key:
                return False
            client = disc.clients.get(key)
            if not client:
                return False
            from ..config.types import EndpointStatus
            if client.state.status not in (EndpointStatus.healthy, EndpointStatus.degraded):
                return False
            # Fast path: exact match in reported models
            if model_name in client.state.models:
                return True
            # Fuzzy path: model name appears as prefix in a reported id
            # (e.g. config "qwen3.6-35b-a3b" vs reported "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
            mn = model_name.lower()
            return any(mn in r.lower() for r in client.state.models)

        # Available local models (privacy=local_only + backend actually reports them)
        available_local: set[str] = {
            name
            for name, m in cfg.models.models.items()
            if m.privacy.value == "local_only"
            and _actually_available(name)
        }

        # Models that appear in any mode definition (llm-mode managed)
        mode_managed: set[str] = set()
        for mode_cfg in cfg.modes.values():
            mode_managed.update(mode_cfg.models)

        # Only compare managed models for mode inference
        active_managed = available_local & mode_managed

        # Match against mode definitions: all mode models available + no extra managed
        # models active that aren't part of this mode
        matched_modes: list[str] = []
        for mode_name, mode_cfg in cfg.modes.items():
            mode_set = set(mode_cfg.models)
            if mode_set == active_managed:
                matched_modes.append(mode_name)
        if not matched_modes:
            matched_modes = ["custom"]

        # Key alias resolutions — first available member wins
        alias_info: dict[str, str | None] = {}
        for alias, members in cfg.models.aliases.items():
            resolved: str | None = None
            for member in members:
                if member not in cfg.models.models:
                    continue
                if disc.get_model_state(member).value in ("available", "degraded"):
                    resolved = member
                    break
            alias_info[alias] = resolved

        # Backend status
        backends: dict[str, Any] = {
            key: {
                "status": c.state.status.value,
                "models": c.state.models,
                "last_poll": c.state.last_poll,
                "inflight_used": c.inflight_used,
                "inflight_capacity": c.max_concurrent,
            }
            for key, c in disc.clients.items()
        }

        return {
            "mode": matched_modes,
            "available_local_models": sorted(available_local),
            "aliases": alias_info,
            "backends": backends,
        }

    @app.get("/routing-table")
    async def routing_table(request: Request) -> dict[str, list[str]]:
        return dict(request.app.state.config.policy.fallback.graph)

    @app.get("/routing-table/{model}")
    async def routing_table_for(model: str, request: Request) -> dict[str, Any]:
        cfg = request.app.state.config.models.models.get(model)
        if not cfg:
            raise HTTPException(404, detail=f"Unknown model: {model}")
        return {
            "model": model,
            "provider": cfg.provider,
            "fallback_chain": request.app.state.router.selector.get_fallback_chain(model),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="Invalid JSON")
        hint_headers: dict[str, str] = {}
        for key in (
            "X-Llm-Relay-Privacy",
            "X-Llm-Relay-Weights",
            "X-Llm-Relay-Require-Tools",
            "X-Llm-Relay-Min-Context",
        ):
            v = request.headers.get(key)
            if v is not None:
                hint_headers[key] = v
        user_agent = request.headers.get("user-agent", "")
        # Explicit X-Llm-Relay-Client header wins; else fall back to a
        # distinctive User-Agent (agent-a); else "unknown". Lets agent-a attribute with
        # zero client-side change while agent-c/agent-b opt in via the header.
        client = resolve_client(request.headers.get("X-Llm-Relay-Client"), user_agent)
        start_ns = time.time_ns()
        is_stream = body.get("stream") is True

        try:
            if is_stream:
                upstream, body_iter, result, cleanup = await request.app.state.router.route_and_forward(
                    request_data=body, headers=hint_headers, stream=True,
                )
            else:
                upstream, result = await request.app.state.router.route_and_forward(
                    request_data=body, headers=hint_headers, stream=False,
                )
        except SaturationError as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=503, streamed=is_stream, error=str(e),
                outcome="saturated", client=client,
            )
            raise HTTPException(
                status_code=503,
                detail={"error": "backend saturated", "backend": e.backend_key,
                        "retry_after_seconds": e.retry_after_seconds},
                headers={"Retry-After": str(max(1, int(e.retry_after_seconds)))},
            )
        except httpx.RequestError as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=is_stream, error=f"Backend network error: {e}",
                outcome="network_error", client=client,
            )
            raise HTTPException(502, detail=f"Backend network error: {e}")
        except HTTPException:
            # route_and_forward raises HTTPException for no-candidates 503.
            # Emit telemetry then re-raise.
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=503, streamed=is_stream, error="No model matches constraints",
                outcome="no_candidate", client=client,
            )
            raise
        except Exception as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=is_stream, error=f"Backend error: {e}",
                outcome="backend_error", client=client,
            )
            raise HTTPException(502, detail=f"Backend error: {e}")

        relay_headers = {
            "X-Llm-Relay-Selected-Model": result.selected_model or "",
            "X-Llm-Relay-Selected-Provider": result.provider_name or "",
        }

        if is_stream:
            media_type = upstream.headers.get("content-type", "text/event-stream")
            upstream_status = upstream.status_code

            async def _tee_and_emit():
                chunks: list[bytes] = []
                try:
                    async for chunk in body_iter:
                        chunks.append(chunk)
                        yield chunk
                finally:
                    text, usage = reassemble_sse(b"".join(chunks))
                    emit_chat_completion(
                        request_body=body, response_body=None, response_text=text, usage=usage,
                        model_resolved=result.selected_model, provider_name=result.provider_name,
                        user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                        status_code=upstream_status, streamed=True,
                        outcome="success" if upstream_status < 400 else "upstream_error",
                        client=client,
                        fell_back=did_fall_back(result.selected_model, (result.decision or {}).get("ranked") or []),
                    )

            # cleanup frees the in-flight slot and closes the upstream
            # connection. Wiring it as the background task guarantees it runs
            # when FastAPI closes the response — including the client-disconnect
            # path, where the response generator might otherwise only be
            # finalized by GC. It's idempotent with the iterator's own finally.
            return StreamingResponse(
                _tee_and_emit(),
                status_code=upstream_status,
                media_type=media_type,
                headers=relay_headers,
                background=BackgroundTask(cleanup),
            )

        try:
            content = upstream.json()
            if isinstance(content, dict):
                content["llm-relay"] = {
                    "selected_model": result.selected_model,
                    "selected_provider": result.provider_name,
                    "decision": result.decision,
                }
        except Exception:
            content = {"raw": upstream.text}
        emit_chat_completion(
            request_body=body,
            response_body=content if isinstance(content, dict) else None,
            response_text=None if isinstance(content, dict) else upstream.text,
            usage=None,
            model_resolved=result.selected_model, provider_name=result.provider_name,
            user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
            status_code=upstream.status_code, streamed=False,
            outcome="success" if upstream.status_code < 400 else "upstream_error",
            client=client,
            fell_back=did_fall_back(result.selected_model, (result.decision or {}).get("ranked") or []),
        )
        return JSONResponse(status_code=upstream.status_code, content=content, headers=relay_headers)

    # Mount MCP at /mcp
    if _mcp_app is not None:
        app.mount("/mcp", _mcp_app)

    # Mount A2A routes (agent card at /.well-known/agent-card.json,
    # JSON-RPC at /a2a/jsonrpc, REST at /a2a/rest)
    if _a2a_card_routes:
        app.routes.extend(_a2a_card_routes)
        app.routes.extend(_a2a_jsonrpc_routes)
        app.routes.extend(_a2a_rest_routes)

    # Prometheus metrics: request/token/fallback counters + pull-based backend
    # gauges, served directly at /metrics (a route, not a mounted sub-app, to
    # avoid the trailing-slash redirect in front of the scrape endpoint).
    if metrics_enabled():
        set_known_routable(set(config.models.models) | set(config.models.aliases))
        register_discovery_collector(discovery)

        @app.get("/metrics")
        def metrics_endpoint() -> Response:
            body, content_type = render_exposition()
            return Response(content=body, media_type=content_type)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("LLM_RELAY_PORT", 8090))
    host = os.environ.get("LLM_RELAY_HOST", "127.0.0.1")
    uvicorn.run("llm_relay.api.app:create_app", host=host, port=port, factory=True)
