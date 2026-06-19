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
from ..routing.keys import compose_backend_key, compose_model_id, resolve_model_id
from ..routing.router import RequestRouter
from ..routing.selector import ModelSelector, RoutingContext
from .instrumentation import (
    _classify_stream_outcome,
    emit_chat_completion,
    reassemble_sse,
    sse_finished,
)
from ..metrics import configure_clients_from_env, did_fall_back, metrics_enabled, register_discovery_collector, render_exposition, resolve_client, set_known_routable


def _resolve_base_url() -> str:
    """Externally-reachable root URL advertised in the MCP config."""
    return os.environ.get("LLM_RELAY_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    if config_dir:
        return Path(config_dir)
    env = os.environ.get("LLM_RELAY_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def _alias_servable_ceiling(cfg: ConfigLoader, disc: DiscoveryManager, alias: str) -> int | None:
    """Largest context window the alias can SERVE right now: the max live window
    among the candidates ``select_chain`` currently yields (named members + open
    fallthrough, already filtered to available + locally-admissible). ``None`` when
    nothing servable is live.

    This drives the honest context advertisement — a client can size a request up
    to this number and the context-fit gate guarantees it routes to a model that
    holds it, instead of being told a down primary's nominal window it can't get.
    """
    sel = ModelSelector(cfg, disc)
    windows = [
        disc.get_live_context_window(c.model) or (cfg.models.models[c.model].context_window or 0)
        for c in sel.select_chain(RoutingContext(requested_model=alias))
    ]
    windows = [w for w in windows if w]
    return max(windows) if windows else None


def _resolve_context_window(cfg: ConfigLoader, disc: DiscoveryManager, name: str) -> int | None:
    """Context window for a model or alias `name`.

    Concrete model: the backend's live ``max_model_len`` (authoritative) when it
    reports one, else the static models.yaml value.

    Alias: the largest context it can SERVE right now — the max live window among
    the models it currently routes to (named members + open fallthrough), via
    ``_alias_servable_ceiling``. This is the honest "size a request up to here"
    number; advertising a down primary's nominal window while a smaller model
    actually serves is the "sized it right, still 503'd" lie. Falls back to the
    primary (first-declared) member's window when nothing servable is live, so the
    advertised capability survives a full-fleet outage.

    Returns None when `name` is neither a known model nor a resolvable alias.
    """
    models = cfg.models.models
    if name in models:
        live = disc.get_live_context_window(name)
        return live if live is not None else models[name].context_window
    members = cfg.models.aliases.get(name)
    if members:
        ceiling = _alias_servable_ceiling(cfg, disc, name)
        if ceiling is not None:
            return ceiling
        for member in members:
            if member in models:
                return _resolve_context_window(cfg, disc, member)
    return None


def _model_entry(
    cfg: ConfigLoader,
    disc: DiscoveryManager,
    model_id: str,
    owned_by: str,
    lookup_name: str | None = None,
) -> dict[str, Any]:
    """One OpenAI ``/v1/models`` entry, enriched with context metadata.

    ``model_id`` is the advertised id (a host-qualified ``provider:model`` for a
    concrete model, or the alias name). ``lookup_name`` is the bare model/alias
    name used to resolve context (defaults to ``model_id``); they differ because
    a concrete model is advertised qualified but its context lives under the
    bare name.

    The OpenAI schema omits context, but vLLM / llama.cpp-style clients discover
    it from ``max_model_len`` / ``context_length`` on the entry. We publish both
    (same value) so a client reading either field gets the right answer; clients
    that don't care ignore the extra keys. Aliases report their primary member's
    context (see :func:`_resolve_context_window`)."""
    entry: dict[str, Any] = {"id": model_id, "object": "model", "owned_by": owned_by}
    ctx = _resolve_context_window(cfg, disc, lookup_name if lookup_name is not None else model_id)
    if ctx is not None:
        entry["context_length"] = ctx
        entry["max_model_len"] = ctx
    return entry


def _build_models_list_payload(cfg: ConfigLoader, disc: DiscoveryManager) -> dict[str, Any]:
    """OpenAI-compatible ``/v1/models`` list: every concrete model and alias,
    each enriched with context metadata so discovery clients can read it from
    the list response (the path most OpenAI-compat resolvers hit first)."""
    data: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, m in cfg.models.models.items():
        if name in seen:
            continue
        seen.add(name)
        # Advertise the host-qualified id so the same model on different hosts is
        # distinguishable; context is still resolved by the bare name.
        data.append(_model_entry(cfg, disc, compose_model_id(m.provider, name), m.provider, lookup_name=name))
    for alias in cfg.models.aliases.keys():
        if alias in seen:
            continue
        seen.add(alias)
        data.append(_model_entry(cfg, disc, alias, "llm-relay-alias"))
    return {"object": "list", "data": data}


def _build_model_card(cfg: ConfigLoader, disc: DiscoveryManager, model: str) -> dict[str, Any] | None:
    """Single OpenAI ``/v1/models/{model}`` card for a model or alias, with
    context metadata. Returns None if `model` is neither a known model nor an
    alias (the route turns that into a 404)."""
    # Accept a bare name or a host-qualified 'provider:model' id (provider
    # validated). Echo the id the caller asked for; resolve context by bare name.
    bare = resolve_model_id(cfg.models.models, model)
    if bare is not None:
        return _model_entry(cfg, disc, model, cfg.models.models[bare].provider, lookup_name=bare)
    if model in cfg.models.aliases:
        return _model_entry(cfg, disc, model, "llm-relay-alias")
    return None


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
            # Isolated backend: reachable only by exact name, never via alias /
            # category fallthrough / open ranking (see selector manual_only). The
            # cockpit shows it; well-behaved auto-pickers should skip it.
            "manual_only": m.manual_only,
        }
        # A deliberately-paused provider reads "paused" (not its discovered
        # status) so clients see it's intentionally out of rotation, not down.
        if disc.is_provider_paused(m.provider):
            out[name]["status"] = "paused"
            client = disc.get_client_for_model(name)
            if client is not None and client.state.paused_until is not None:
                out[name]["paused_until"] = client.state.paused_until
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
        # `context_window` is the live-servable CEILING (max live window among the
        # alias's currently-routable candidates, via _resolve_context_window ->
        # _alias_servable_ceiling) — the number a client can safely size a request
        # up to. It tracks the live fleet rather than a down primary's nominal
        # window, so a client never sizes to context the alias cannot actually hold
        # (the "sized it right, still 503'd" lie). Size the PROMPT as chars/3 <=
        # context_window; max_tokens is an output ceiling, clamped to the chosen
        # model's headroom at forward time (see _clamp_max_tokens), not counted
        # toward eligibility (see _estimate_prompt_tokens).
        alias_info[alias] = {
            "members": members_list,
            "current": current,
            "context_window": _resolve_context_window(cfg, disc, alias),
        }
    out["alias_info"] = alias_info
    return out


def create_app(config_dir: str | Path | None = None) -> FastAPI:
    cfg_path = _resolve_config_dir(config_dir)
    config = ConfigLoader(config_dir=cfg_path)
    config.load()
    # Load deployment-specific client attribution (known client labels + UA
    # patterns) from the env; a no-op leaving generic defaults if unset. Done
    # before routing so resolve_client is correct for telemetry and metrics.
    configure_clients_from_env()
    discovery = DiscoveryManager()
    # Seed served-name overrides so availability correlates a config model with the
    # id its backend actually reports in /v1/models (e.g. a GGUF filename).
    for _name, _m in config.models.models.items():
        if _m.served_model_name:
            discovery.served_names[_name] = _m.served_model_name
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
            # Restore any persisted maintenance pauses now that every backend is
            # registered (discovery.clients is populated). Doing this at
            # create_app time would be a no-op -- clients are empty there.
            for _prov, _info in config.load_paused_providers().items():
                discovery.pause_provider(_prov, _info.get("until"), _info.get("reason"))
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
    async def available_models(request: Request, response: Response) -> dict[str, Any]:
        # Deprecated alias of /v1/available-models (the canonical, OpenAI-namespaced
        # path MCP and clients use). Kept working so no caller breaks; RFC 8594
        # headers point them at the successor.
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = '</v1/available-models>; rel="successor-version"'
        return await _available(request)

    @app.get("/v1/available-models")
    async def available_models_v1(request: Request) -> dict[str, Any]:
        return await _available(request)

    @app.get("/v1/models")
    async def list_models_openai(request: Request) -> dict[str, Any]:
        return _build_models_list_payload(
            request.app.state.config, request.app.state.discovery
        )

    @app.get("/v1/models/{model}")
    async def get_model_openai(model: str, request: Request) -> dict[str, Any]:
        # OpenAI per-model card. Many OpenAI-compat discovery clients probe this
        # before falling back to the list; today its absence (404) forced them
        # onto stale/default context values. Serve a card with context metadata.
        card = _build_model_card(
            request.app.state.config, request.app.state.discovery, model
        )
        if card is None:
            raise HTTPException(404, detail=f"Unknown model: {model}")
        return card

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
            # (e.g. config "model-x" vs reported "Model-X-Instruct-UD-Q4_K_XL.gguf")
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

        # Backend status. Surface the maintenance "paused" flag -- routed through
        # is_provider_paused so an expired pause is honored (and healed) here too,
        # not just on /v1/available-models -- so operators and the mi100 watchdog
        # can tell a deliberate pause from a real outage.
        backends: dict[str, Any] = {}
        for key, c in disc.clients.items():
            entry: dict[str, Any] = {
                "status": c.state.status.value,
                "models": c.state.models,
                "last_poll": c.state.last_poll,
                "inflight_used": c.inflight_used,
                "inflight_capacity": c.max_concurrent,
            }
            if disc.is_provider_paused(c.provider_name):
                entry["paused"] = True
                entry["paused_until"] = c.state.paused_until
                entry["paused_reason"] = c.state.paused_reason
            backends[key] = entry

        return {
            "mode": matched_modes,
            "available_local_models": sorted(available_local),
            "aliases": alias_info,
            "backends": backends,
        }

    @app.post("/admin/pause")
    async def admin_pause(request: Request) -> dict[str, Any]:
        """Put a provider into maintenance ("paused"): the selector skips its
        backends without tripping the circuit breaker, and it reads "paused"
        (not "down"). Body: {"provider": str, "until": ISO8601|null, "reason":
        str|null}. Persisted (paused-providers.json) so it survives a restart.
        404 if the provider is not configured. Used by the Reno fleet dashboard
        scheduler; the dashboard web app never calls this."""
        cfg = request.app.state.config
        disc = request.app.state.discovery
        body = await request.json()
        provider = body.get("provider")
        if provider not in cfg.providers:
            raise HTTPException(404, detail=f"Unknown provider: {provider}")
        until, reason = body.get("until"), body.get("reason")
        disc.pause_provider(provider, until, reason)
        persisted = cfg.load_paused_providers()
        persisted[provider] = {"until": until, "reason": reason}
        cfg.save_paused_providers(persisted)
        return {"ok": True, "provider": provider, "paused": disc.is_provider_paused(provider)}

    @app.post("/admin/resume")
    async def admin_resume(request: Request) -> dict[str, Any]:
        """Take a provider out of maintenance. Body: {"provider": str}. Clears
        the persisted pause too. 404 if the provider is not configured."""
        cfg = request.app.state.config
        disc = request.app.state.discovery
        body = await request.json()
        provider = body.get("provider")
        if provider not in cfg.providers:
            raise HTTPException(404, detail=f"Unknown provider: {provider}")
        disc.resume_provider(provider)
        persisted = cfg.load_paused_providers()
        persisted.pop(provider, None)
        cfg.save_paused_providers(persisted)
        return {"ok": True, "provider": provider, "paused": disc.is_provider_paused(provider)}

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
            "X-Llm-Relay-Require-Tools",
            "X-Llm-Relay-Min-Context",
        ):
            v = request.headers.get(key)
            if v is not None:
                hint_headers[key] = v
        user_agent = request.headers.get("user-agent", "")
        # Explicit X-Llm-Relay-Client header wins; else fall back to a configured
        # distinctive User-Agent pattern; else "unknown". The known-client set and
        # UA patterns are deployment-configured via the environment (see
        # metrics.configure_clients_from_env), so a client with a distinctive UA
        # can be attributed with zero client-side change while others opt in via
        # the header.
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
                exc: BaseException | None = None
                first_chunk_ns: int | None = None
                try:
                    async for chunk in body_iter:
                        if first_chunk_ns is None:
                            first_chunk_ns = time.time_ns()
                        chunks.append(chunk)
                        yield chunk
                except BaseException as e:
                    # Capture HOW the stream ended so the outcome is honest, then
                    # always re-raise: a swallowed CancelledError/GeneratorExit would
                    # strand the cleanup that frees the in-flight slot.
                    exc = e
                    raise
                finally:
                    raw = b"".join(chunks)
                    text, usage = reassemble_sse(raw)
                    # Outcome reflects the ACTUAL termination, not just the initial
                    # status: a 200 that stalls mid-stream is not a success. Only
                    # synchronous calls here — under GeneratorExit we must not await.
                    outcome = _classify_stream_outcome(upstream_status, exc, sse_finished(raw))
                    # End-to-end TTFT (first chunk ~= first byte, includes routing);
                    # None when no chunk ever flowed.
                    ttft_ns = (first_chunk_ns - start_ns) if first_chunk_ns is not None else None
                    emit_chat_completion(
                        request_body=body, response_body=None, response_text=text, usage=usage,
                        model_resolved=result.selected_model, provider_name=result.provider_name,
                        user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                        status_code=upstream_status, streamed=True,
                        outcome=outcome,
                        client=client,
                        fell_back=did_fall_back(result.selected_model, (result.decision or {}).get("ranked") or []),
                        ttft_ns=ttft_ns,
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
