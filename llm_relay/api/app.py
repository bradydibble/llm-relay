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
from fastapi.responses import JSONResponse, StreamingResponse

from ..config.loader import ConfigLoader
from ..discovery.manager import DiscoveryManager
from ..routing.router import RequestRouter
from .instrumentation import emit_chat_completion, reassemble_sse


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
        out[name] = {
            "provider": m.provider,
            "class": m.class_name,
            "status": status.value,
            "context_window": m.context_window,
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
        cw = cfg.models.models[current].context_window if current else None
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
                key_parts = [provider_name]
                if port:
                    key_parts.append(str(port))
                if path:
                    key_parts.append(path.strip("/"))
                key = ":".join(key_parts)
                await discovery.register_backend(
                    key=key,
                    provider_name=provider_name,
                    base_url=base,
                    models_hint=names,
                    health_endpoint=provider.health_endpoint,
                    poll_interval=provider.poll_interval,
                    circuit_breaker=provider.circuit_breaker,
                    timeout=provider.health_check_timeout,
                )
        yield
        await discovery.shutdown()

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
        start_ns = time.time_ns()
        result = await request.app.state.router.route_request(body, hint_headers)
        if not result.success:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=result.selected_model, provider_name=result.provider_name,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=503, streamed=False, error=str(result.error),
            )
            raise HTTPException(503, detail={"error": result.error, "decision": result.decision})

        relay_headers = {
            "X-Llm-Relay-Selected-Model": result.selected_model or "",
            "X-Llm-Relay-Selected-Provider": result.provider_name or "",
        }

        if body.get("stream") is True:
            try:
                upstream, body_iter = await request.app.state.router.stream_request(
                    result.backend_url, result.selected_model, body
                )
            except httpx.RequestError as e:
                emit_chat_completion(
                    request_body=body, response_body=None, response_text=None, usage=None,
                    model_resolved=result.selected_model, provider_name=result.provider_name,
                    user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                    status_code=502, streamed=True, error=f"Backend network error: {e}",
                )
                raise HTTPException(502, detail=f"Backend network error: {e}")
            except Exception as e:
                emit_chat_completion(
                    request_body=body, response_body=None, response_text=None, usage=None,
                    model_resolved=result.selected_model, provider_name=result.provider_name,
                    user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                    status_code=502, streamed=True, error=f"Backend error: {e}",
                )
                raise HTTPException(502, detail=f"Backend error: {e}")
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
                    )

            return StreamingResponse(
                _tee_and_emit(),
                status_code=upstream_status,
                media_type=media_type,
                headers=relay_headers,
            )

        try:
            upstream = await request.app.state.router.forward_request(
                result.backend_url, result.selected_model, body
            )
        except httpx.RequestError as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=result.selected_model, provider_name=result.provider_name,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=False, error=f"Backend network error: {e}",
            )
            raise HTTPException(502, detail=f"Backend network error: {e}")
        except Exception as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=result.selected_model, provider_name=result.provider_name,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=False, error=f"Backend error: {e}",
            )
            raise HTTPException(502, detail=f"Backend error: {e}")
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
        )
        return JSONResponse(status_code=upstream.status_code, content=content, headers=relay_headers)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("LLM_RELAY_PORT", 8090))
    host = os.environ.get("LLM_RELAY_HOST", "127.0.0.1")
    uvicorn.run("llm_relay.api.app:create_app", host=host, port=port, factory=True)
