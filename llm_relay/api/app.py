"""FastAPI application for llm-relay."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from ..config.loader import ConfigLoader
from ..config.types import ModelConfig, ModelStatus, NoBackendAvailableError, Privacy, SaturationError
from ..discovery.manager import DiscoveryManager
from ..routing.keys import compose_backend_key, compose_model_id, resolve_model_id
from ..routing.router import ContextLengthExceededError, RequestRouter
from .instrumentation import (
    _classify_stream_outcome,
    emit_chat_completion,
    reassemble_sse,
    sse_finished,
)
from ..metrics import configure_clients_from_env, did_fall_back, metrics_enabled, register_discovery_collector, render_exposition, resolve_client, set_known_routable
from ..logbuffer import install_log_buffer
from ..scheduler import AdmissionController
from ..jobs import JobStore
from ..jobworker import run_worker
from ..health import L2HealthProbe
from ..degeneracy import is_degenerate, degeneracy_score
from ..config_drift import ConfigDriftDetector


_KEEPALIVE_INTERVAL_S = 15.0


def _mirror_reasoning(payload: dict) -> bool:
    """Normalize reasoning field names so EITHER `reasoning` or `reasoning_content`
    is always present on each choice (`message` for non-stream, `delta` for stream).

    Why this lives in the relay and NOT in the serve (deliberate, durable):
    vLLM RENAMED the field `reasoning_content` -> `reasoning` and deprecated the old
    name (docs.vllm.ai reasoning_outputs: "reasoning used to be called
    reasoning_content ... directly replace"). There is NO serve flag to emit the old
    name — reverting it would mean forking vLLM and re-applying the patch on every
    upgrade, the exact treadmill we avoid. Meanwhile a fleet is often MIXED-version:
    newer vLLM builds emit `reasoning`, older builds emit `reasoning_content`, and
    clients (pi/zed/OpenAI SDKs) variously key on one or the other. So the relay —
    the one point that spans the whole fleet — mirrors
    BOTH directions: whichever field the serve emits, the other is copied in. This is
    non-breaking (a client reading either name works) and immune to vLLM renaming the
    field in future patches. Mutates in place; returns True iff it changed anything;
    tolerant of any shape (no-op on surprises)."""
    changed = False
    try:
        for ch in payload.get("choices", []) or []:
            if not isinstance(ch, dict):
                continue
            for key in ("message", "delta"):
                node = ch.get(key)
                if not isinstance(node, dict):
                    continue
                r = node.get("reasoning")
                rc = node.get("reasoning_content")
                if r is not None and rc is None:
                    node["reasoning_content"] = r
                    changed = True
                elif rc is not None and r is None:
                    node["reasoning"] = rc
                    changed = True
    except Exception:
        pass
    return changed


def _mirror_reasoning_sse_frame(frame: str) -> str:
    """Apply ``_mirror_reasoning`` to any ``data: {json}`` line in a single,
    COMPLETE SSE frame. Keepalive comments (``: ...``), ``data: [DONE]``, and any
    non-JSON line pass through byte-identical; a frame we do not change is returned
    unchanged (never re-serialized). Frame boundaries (the trailing blank line) are
    handled by the caller — this only rewrites line content."""
    if "reasoning" not in frame:  # cheap fast-path: nothing to mirror
        return frame
    lines = frame.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict) and _mirror_reasoning(obj):
            lines[i] = "data: " + json.dumps(obj, separators=(",", ":"))
            changed = True
    return "\n".join(lines) if changed else frame


async def _sse_stream_keepalive(body_iter, media_type, interval):
    """Yield ``(payload, is_keepalive)`` from an upstream byte iterator, emitting
    an SSE comment frame (``: ka``) whenever no upstream chunk arrives within
    ``interval`` seconds.

    Large-context ornith-397b requests spend tens of seconds in prefill before
    the first token (observed: 120k ctx -> ~90s TTFT), during which the streaming
    path is otherwise silent. That silent gap trips Cloudflare's ~100s edge-idle
    timeout and client read timeouts, killing the connection before any token
    flows. SSE comment lines are ignored by all SSE/OpenAI clients, so they keep
    the connection warm without corrupting the stream or the token content.

    Only applied to ``text/event-stream`` responses; any other media type passes
    through untouched. Keepalive payloads are relay-injected and are NOT part of
    the upstream body, so callers must not fold them into reassembly/usage.
    """
    if not (media_type or "").startswith("text/event-stream"):
        async for chunk in body_iter:
            yield chunk, False
        return
    ait = body_iter.__aiter__()
    pending = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(ait.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if pending in done:
                nxt, pending = pending, None
                try:
                    chunk = nxt.result()
                except StopAsyncIteration:
                    return
                yield chunk, False
            else:
                yield b": ka\n\n", True
    finally:
        if pending is not None:
            pending.cancel()
            with suppress(BaseException):
                await pending


def _backpressure_response(status_code, err_type, message, retry_after_seconds, extra=None):
    """Well-formed backpressure response. The error goes at the TOP LEVEL
    (``{"error": {...}}``), never nested under FastAPI's ``detail`` — clients
    (pi, Paseo, OpenAI SDKs) parse the top-level shape and otherwise see the
    rejection as "no body". A real ``Retry-After`` header is always set.

    Status: **429** for slot saturation (too many concurrent requests on a
    reachable backend — the caller should back off and retry the SAME model),
    **503** only for a transient backend-down/paused gap. A 503 for saturation
    was the bug: it reads as a server fault, not backpressure.
    """
    err = {"message": message, "type": err_type, "code": err_type,
           "retry_after_seconds": retry_after_seconds}
    if extra:
        err.update(extra)
    return JSONResponse(
        status_code=status_code,
        content={"error": err},
        headers={"Retry-After": str(max(1, int(retry_after_seconds))),
                 "X-Llm-Relay-Error": err_type},
    )


def _resolve_base_url() -> str:
    """Externally-reachable root URL advertised in the MCP config."""
    return os.environ.get("LLM_RELAY_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


def _job_visible(job, principal, auth_enabled: bool) -> bool:
    """Jobs are principal-scoped on the auth listener: a caller sees only its
    own jobs unless it carries the admin scope (the trusted listener's implicit
    principal does). Auth disabled = legacy open behavior."""
    if not auth_enabled:
        return True
    pid = getattr(principal, "id", "anonymous")
    scopes = list(getattr(principal, "scopes", []) or [])
    return job.principal == pid or "admin" in scopes


def _clamp_privacy(principal, auth_enabled: bool, hint_headers: dict[str, str]) -> None:
    """Privacy ceiling: only principals carrying the ``cloud`` scope may pass
    ``cloud_ok`` upstream (trusted-listener traffic carries it implicitly).
    With auth disabled, legacy open-deployment behavior is preserved."""
    if not auth_enabled:
        return
    scopes = list(getattr(principal, "scopes", []) or [])
    if "cloud" in scopes:
        return
    if hint_headers.get("X-Llm-Relay-Privacy") == "cloud_ok":
        hint_headers["X-Llm-Relay-Privacy"] = "local_only"


def _ownership_value(cfg, provider_name: str) -> str:
    """Ownership string for *provider_name*, failing closed to ``third_party``
    when the provider cannot be resolved — mirrors ``ModelSelector._ownership_of``
    so the discovery surface never advertises a model as safer than routing will
    actually treat it."""
    provider = cfg.providers.get(provider_name)
    return provider.ownership.value if provider else "third_party"


def _clamp_confidentiality(principal, auth_enabled: bool, hint_headers: dict[str, str]) -> None:
    """Confidentiality ceiling: only principals carrying the ``third_party``
    scope may declare a workload ``non_confidential`` and thereby reach hardware
    CIQ does not own.

    The declaration itself is the operator's to make — the relay cannot inspect a
    prompt and decide whether it contains sales data or kernel patches. What it
    CAN do is bound who is allowed to make the claim at all, so a misconfigured
    or copy-pasted agent cannot unilaterally route CIQ-proprietary work onto
    borrowed metal. A caller without the scope is silently clamped back to
    ``confidential`` (fail-closed) rather than rejected, so an over-broad header
    degrades to the safe pool instead of breaking the request.

    Trusted-listener traffic (the keyless :8090 tailnet listener) carries the
    scope implicitly, and with auth disabled the legacy open-deployment
    behavior is preserved — matching ``_clamp_privacy`` exactly.
    """
    if not auth_enabled:
        return
    scopes = list(getattr(principal, "scopes", []) or [])
    if "third_party" in scopes:
        return
    if hint_headers.get("X-Llm-Relay-Confidentiality") == "non_confidential":
        hint_headers["X-Llm-Relay-Confidentiality"] = "confidential"


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    if config_dir:
        return Path(config_dir)
    env = os.environ.get("LLM_RELAY_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def _alias_current_member(cfg: ConfigLoader, disc: DiscoveryManager, alias: str) -> str | None:
    """The member an alias routes a normally-sized request to RIGHT NOW: the first
    declared member that discovery reports available/degraded, else the first
    declared configured member (so the answer survives a full-fleet outage).

    This is the model whose context window the alias should advertise — see
    ``_resolve_context_window``. It matches ``select_best`` for a request with no
    context floor, and the ``current`` field in ``_build_available_payload``.
    """
    members = cfg.models.aliases.get(alias) or []
    for m in members:
        if m in cfg.models.models and disc.get_model_state(m).value in ("available", "degraded"):
            return m
    for m in members:
        if m in cfg.models.models:
            return m
    return None


def _resolve_context_window(cfg: ConfigLoader, disc: DiscoveryManager, name: str) -> int | None:
    """Context window for a model or alias `name`.

    Concrete model: the backend's live ``max_model_len`` (authoritative) when it
    reports one, else the static models.yaml value.

    Alias: the window of the member it routes a normally-sized request to RIGHT
    NOW (``_alias_current_member`` — first available member, else first declared),
    NOT the fleet-max ceiling across its open-fallthrough tail. This is the number
    a client's autocompaction must key off: a `subagent` alias fronting the 64K 9B
    must advertise 65536 so the harness compacts to stay on the fast model, not the
    262K a big fallthrough model *could* serve — advertising that ceiling let a
    66K prompt sail past pi's autocompact straight into the 9B's wall (2026-07-07).
    A too-big prompt still escalates up the chain at request time (the selector's
    context-fit gate + router's overflow backstop); advertisement tracks the
    everyday routing target, escalation handles the exception.

    Returns None when `name` is neither a known model nor a resolvable alias.
    """
    models = cfg.models.models
    if name in models:
        live = disc.get_live_context_window(name)
        return live if live is not None else models[name].context_window
    if name in cfg.models.aliases:
        current = _alias_current_member(cfg, disc, name)
        if current is not None:
            return _resolve_context_window(cfg, disc, current)
    return None


def _model_entry(
    cfg: ConfigLoader,
    disc: DiscoveryManager,
    model_id: str,
    owned_by: str,
    lookup_name: str | None = None,
    members: list[str] | None = None,
) -> dict[str, Any]:
    """One OpenAI ``/v1/models`` entry, enriched with context metadata.

    ``members`` (aliases only) names the models the alias can route to, so the
    capabilities block can be their intersection; None means "this is a concrete
    model, use its own config".

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
        # The schema OpenAI-compat pickers (ciq-harness among them) actually read.
        # Publishing context ONLY as context_length/max_model_len left that client
        # filling in a 131072 default for every model - overstating the 32k models
        # and halving the 262k ones. One value, every spelling a client looks for.
        entry["limit"] = {"context": ctx}
    m = cfg.models.models.get(lookup_name if lookup_name is not None else model_id)
    if m is not None and m.description:
        entry["description"] = m.description
    caps = _capabilities_block(cfg, members if members is not None
                               else [lookup_name if lookup_name is not None else model_id])
    if caps is not None:
        entry["capabilities"] = caps
    return entry


def _capabilities_block(cfg: ConfigLoader, names: list[str]) -> dict[str, bool] | None:
    """Capability claims for a model or alias, as booleans a picker can act on.

    For an alias, the INTERSECTION across its members: an alias only advertises
    what EVERY member delivers, because the client cannot know which member will
    answer. If `main` can fall through to a model without tool support, then
    `main` does not support tools - advertising the union is how a tool-bearing
    request ends up "succeeding" with empty content.

    A capability listed here is a routable claim (require_tools filters on
    tool_use), so publishing it and enforcing it use the same source of truth.
    Returns None when no named member exists in config (unknown/discovered-only),
    on the additive-fields principle: absent, not fabricated."""
    configs = [cfg.models.models[n] for n in names if n in cfg.models.models]
    if not configs:
        return None
    def every(cap: str) -> bool:
        return all(cap in c.capabilities for c in configs)
    return {
        "toolcall": every("tool_use"),
        "reasoning": every("reasoning"),
        "structured_output": every("structured_output"),
    }


def _build_models_list_payload(cfg: ConfigLoader, disc: DiscoveryManager) -> dict[str, Any]:
    """OpenAI-compatible ``/v1/models`` list, enriched with context metadata so
    discovery clients can read it from the list response (the path most
    OpenAI-compat resolvers hit first).

    Only models the relay can serve RIGHT NOW (status available/degraded per
    discovery, same authority as ``/available-models``) are advertised; an
    alias stays listed while at least one member is servable. Model pickers
    (Open WebUI etc.) therefore show the live fleet, not the config file.

    Fail-open-when-blind: if discovery reports NOTHING servable (typically a
    just-restarted relay before its first poll, or a total fleet outage), the
    full configured list is returned instead — an empty list would tell
    clients the fleet doesn't exist, which is a worse lie than a stale one.
    ``/available-models`` remains the full config-with-status view."""
    usable = {ModelStatus.available, ModelStatus.degraded}
    states = {name: disc.get_model_state(name) for name in cfg.models.models}
    filtering = any(s in usable for s in states.values())
    data: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, m in cfg.models.models.items():
        if name in seen:
            continue
        seen.add(name)
        if filtering and states[name] not in usable:
            continue
        # Advertise the host-qualified id so the same model on different hosts is
        # distinguishable; context is still resolved by the bare name.
        data.append(_model_entry(cfg, disc, compose_model_id(m.provider, name), m.provider, lookup_name=name))
    for alias, members in cfg.models.aliases.items():
        if alias in seen:
            continue
        seen.add(alias)
        if filtering and not any(states.get(member) in usable for member in members):
            continue
        data.append(_model_entry(cfg, disc, alias, "llm-relay-alias", members=list(members)))
    return {"object": "list", "data": data}


def _build_model_card(cfg: ConfigLoader, disc: DiscoveryManager, model: str) -> dict[str, Any] | None:
    """Single OpenAI ``/v1/models/{model}`` card for a model or alias, with
    context metadata. Returns None if `model` is neither a known model nor an
    alias (the route turns that into a 404)."""
    # Accept a bare name or a host-qualified 'provider:model' id (provider
    # validated). Echo the id the caller asked for; resolve context by bare name.
    bare = resolve_model_id(cfg.models.models, model)
    if bare is not None:
        entry = _model_entry(cfg, disc, model, cfg.models.models[bare].provider, lookup_name=bare)
        # Flag runtime-discovered models so a client can tell an unmanaged
        # bake-off model from a configured one (additive; absent otherwise).
        if cfg.models.models[bare].discovered:
            entry["discovered"] = True
        return entry
    if model in cfg.models.aliases:
        return _model_entry(cfg, disc, model, "llm-relay-alias",
                            members=list(cfg.models.aliases[model]))
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
            # Who owns the metal this model runs on, inherited from its provider,
            # plus the consequence a client actually has to act on: a model on
            # third-party hardware is unreachable unless the request declares the
            # workload non-confidential. Surfaced so pickers can filter up front
            # instead of discovering it as a 503.
            "ownership": _ownership_value(cfg, m.provider),
            "requires_non_confidential": _ownership_value(cfg, m.provider) != "ciq_owned",
            "port": m.port,
            "path": m.path,
            # Isolated backend: reachable only by exact name, never via alias /
            # category fallthrough / open ranking (see selector manual_only). The
            # cockpit shows it; well-behaved auto-pickers should skip it.
            "manual_only": m.manual_only,
        }
        if m.description:
            out[name]["description"] = m.description
        # Runtime-discovered (found on a provider's discover_ports, not configured):
        # additive flag so the cockpit can distinguish ad-hoc bake-off models.
        if m.discovered:
            out[name]["discovered"] = True
        # Variant grouping (plan 2): additive, present only when declared.
        if m.logical:
            out[name]["logical"] = m.logical
        if m.quant:
            out[name]["quant"] = m.quant
        # A deliberately-paused provider reads "paused" (not its discovered
        # status) so clients see it's intentionally out of rotation, not down.
        if disc.is_provider_paused(m.provider):
            out[name]["status"] = "paused"
            client = disc.get_client_for_model(name)
            if client is not None and client.state.paused_until is not None:
                out[name]["paused_until"] = client.state.paused_until
    out["aliases"] = dict(cfg.models.aliases)
    # Variant registry (plan 2): logical models -> their variant names, and the
    # mutually-exclusive groups (models sharing a served provider+port). Additive.
    out["logical_models"] = {k: list(v) for k, v in cfg.models.logical_models.items()}
    out["exclusivity_groups"] = [list(g) for g in cfg.models.exclusivity_groups]
    # Enriched per-alias metadata so clients can show context_window etc. for
    # aliases (which are otherwise just names). `current` is a display
    # approximation: the first member that discovery reports as available /
    # degraded, falling back to the first declared member. The selector applies
    # additional filters (privacy, min_context, require_tools) at request time,
    # so the actually-routed model may differ.
    alias_info: dict[str, Any] = {}
    for alias, members in cfg.models.aliases.items():
        members_list = list(members)
        current = _alias_current_member(cfg, disc, alias)
        # `context_window` is the window of `current` — the member this alias routes
        # a normally-sized request to right now (see _resolve_context_window). A
        # client keys its autocompaction off this number, so it must be the everyday
        # target's window, NOT the fleet-max a big fallthrough model could serve
        # (advertising that let a 66K prompt overrun the 64K 9B on 2026-07-07). A
        # prompt too big for `current` still escalates up the chain at request time
        # (selector context-fit gate + router overflow backstop). Size the PROMPT
        # under context_window; max_tokens is an output ceiling clamped per-candidate
        # at forward time (see _clamp_max_tokens), not counted toward eligibility.
        alias_info[alias] = {
            "members": members_list,
            "current": current,
            "context_window": _resolve_context_window(cfg, disc, alias),
        }
    out["alias_info"] = alias_info
    return out


def _reconcile_discovered(
    config: ConfigLoader,
    discovery: DiscoveryManager,
    discover_keys: set[str],
    key_meta: dict[str, tuple[str, int]],
) -> None:
    """Sync the runtime-discovered model registry with what the discover-port
    backends currently report.

    ``discover_keys`` are the discovery-client keys registered for the providers'
    ``discover_ports``; ``key_meta`` maps each to its ``(provider_name, port)``.

    For every model id a discover-port backend reports in ``/v1/models`` that is
    NOT already a statically-configured model, register a ``ModelConfig`` for it
    so the selector can route to it by exact name (it carries ``manual_only`` /
    ``discovered`` so it never joins an alias tail or open ranking). Models that
    stop being reported are dropped. Synchronous and side-effecting on
    ``config``/``discovery`` so the lifespan reconcile task — and unit tests —
    can call it directly.
    """
    # 1. What the discover ports currently serve, keyed served_name -> backend key.
    #    A served id that collides with a STATIC model is left to that model
    #    (never shadow a configured entry); a previously-discovered id is fair game.
    seen: dict[str, str] = {}
    for key in discover_keys:
        client = discovery.clients.get(key)
        if client is None:
            continue
        for sid in client.state.models:
            existing = config.models.models.get(sid)
            if existing is not None and not existing.discovered:
                continue  # static model takes precedence — don't shadow it
            seen[sid] = key

    # 2. Register anything newly seen as a discovered, name-only-routable model.
    #    provider/port are set so compose_backend_key(provider, port, "") resolves
    #    back to the discover key the model is served on (the selector keys off it).
    for sid, key in seen.items():
        provider_name, port = key_meta[key]
        if sid not in config.models.models:
            config.models.models[sid] = ModelConfig(
                provider=provider_name,
                port=port,
                served_model_name=sid,
                context_window=None,
                capabilities=[],
                tags=["discovered"],
                use_cases={},
                manual_only=True,
                discovered=True,
                privacy=Privacy.local_only,
            )
        # Point availability / routing at the discover backend serving it.
        discovery.model_to_client[sid] = key
        discovery.served_names.setdefault(sid, sid)

    # 3. Drop discovered models whose port no longer reports them.
    for name in [
        n for n, m in config.models.models.items() if m.discovered and n not in seen
    ]:
        config.models.models.pop(name, None)
        discovery.model_to_client.pop(name, None)
        discovery.served_names.pop(name, None)


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
            # Register a bare polling client for each provider discover_port that
            # isn't already a configured model's port. These ports carry no
            # models_hint -- whatever they report in /v1/models is reconciled into
            # the registry (as discovered, name-only-routable models) by the task
            # below, so an ad-hoc / bake-off model on an unmanaged port is picked
            # up automatically instead of making the host read "down".
            discover_keys: set[str] = set()
            key_meta: dict[str, tuple[str, int]] = {}
            discover_poll_intervals: list[int] = []
            for provider_name, provider in config.providers.items():
                if not provider.enabled or not provider.discover_ports:
                    continue
                discover_poll_intervals.append(provider.poll_interval)
                for port in provider.discover_ports:
                    key = compose_backend_key(provider_name, port, "")
                    if key in discovery.clients:
                        continue  # a static model already polls this port
                    await discovery.register_backend(
                        key=key,
                        provider_name=provider_name,
                        base_url=f"{provider.base_url.rstrip('/')}:{port}",
                        models_hint=[],
                        health_endpoint=provider.health_endpoint,
                        poll_interval=provider.poll_interval,
                        circuit_breaker=provider.circuit_breaker,
                        timeout=provider.health_check_timeout,
                        max_concurrent=provider.max_concurrent,
                    )
                    discover_keys.add(key)
                    key_meta[key] = (provider_name, port)
            # Restore any persisted maintenance pauses now that every backend is
            # registered (discovery.clients is populated). Doing this at
            # create_app time would be a no-op -- clients are empty there.
            for _prov, _info in config.load_paused_providers().items():
                discovery.pause_provider(_prov, _info.get("until"), _info.get("reason"))
            # Run the discover-port reconcile on the same cadence as polling so a
            # model appearing/disappearing on a discover port is reflected within a
            # poll cycle. The first poll hasn't happened yet, so the loop sleeps
            # before its first pass. Appended to discovery._tasks so the existing
            # shutdown cancels it with the poll loops.
            if discover_keys:
                interval = max(5, min(discover_poll_intervals))

                async def _reconcile_loop() -> None:
                    while True:
                        await asyncio.sleep(interval)
                        _reconcile_discovered(config, discovery, discover_keys, key_meta)

                discovery._tasks.append(asyncio.create_task(_reconcile_loop()))
            # Async job worker (plan 4 slice 2): reconcile any jobs left running by
            # a crash (-> interrupted, never silently re-run), then run until shutdown.
            app.state.job_store.reconcile_on_start()
            _job_stop = asyncio.Event()
            _job_task = asyncio.create_task(run_worker(app.state.job_store, router, _job_stop))
            # L2 inference health probe: background loop that sends a tiny
            # completion to each healthy backend every 30s with a 10s hard
            # timeout. Catches wedged generation slots that /v1/models can't
            # detect (process alive, slot stuck). On 2 consecutive failures,
            # marks the backend degraded and excludes it from routing.
            _l2_probe = L2HealthProbe(discovery, config)
            _l2_task = _l2_probe.start()
            _drift_detector = ConfigDriftDetector(discovery, config)
            _drift_task = _drift_detector.start()
            try:
                yield
            finally:
                _job_stop.set()
                await _l2_probe.stop()
                await _drift_detector.stop()
                try:
                    await asyncio.wait_for(_job_task, timeout=10)
                except Exception:
                    pass
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
    app.state.admission = AdmissionController()
    # Durable async job store (plan 4 slice 2); the worker is started in lifespan.
    app.state.job_store = JobStore(cfg_path / "jobs.json")

    # Per-user API-key auth (a no-op when disabled). Installed here so it wraps
    # every route, including the MCP mount.
    from .middleware import install_auth_middleware
    install_auth_middleware(app)

    # In-memory log buffer for /logs and /logs/stream (plan 7), tailed by the cockpit.
    app.state.log_buffer = install_log_buffer()

    async def _available(request: Request) -> dict[str, Any]:
        return _build_available_payload(
            request.app.state.config, request.app.state.discovery
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        disc = request.app.state.discovery
        # /health is auth-exempt (liveness probes). When auth is on, return a
        # minimal body so a keyless caller cannot read backend topology here.
        if request.app.state.config.auth.enabled:
            return {"status": "ok"}
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

        # Runtime-discovered models (found on a provider's discover_ports, not in
        # models.yaml). Surfaced as a distinct list so an operator can tell an
        # ad-hoc bake-off model from a configured one; additive (empty when none).
        discovered_models = sorted(
            name for name, m in cfg.models.models.items() if m.discovered
        )

        return {
            "mode": matched_modes,
            "available_local_models": sorted(available_local),
            "discovered_models": discovered_models,
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

    # --- Key lifecycle over HTTP (admin scope enforced by the middleware). ---
    # API minting is deliberately scope-less: admin-scoped keys are mintable
    # only via the on-box CLI, so a leaked admin key can manage users but
    # cannot quietly create more admins.
    from ..audit import audit as _audit
    from ..auth import (
        add_key_record,
        load_key_records,
        load_keys as _load_keys,
        revoke_hash,
        update_key_scopes,
    )

    _keys_path = cfg_path / "api_keys.yaml"

    def _reload_principals() -> None:
        config.auth.principals_by_hash = _load_keys(_keys_path)

    def _acting(request: Request) -> str:
        return getattr(getattr(request.state, "principal", None), "id", "?")

    def _owner_email(body: dict[str, Any]) -> str:
        email = str(body.get("owner_email") or "").strip().lower()
        local, separator, domain = email.partition("@")
        if not local or separator != "@" or not domain or "." not in domain:
            raise HTTPException(400, detail="valid owner email required")
        return email

    def _owner_key_view(key_hash: str, record: dict[str, Any]) -> dict[str, Any]:
        """Metadata safe to show to the owner or a portal administrator.

        The key hash and plaintext bearer value deliberately never cross this
        API. The short hash prefix exists only to identify a record for a later
        revoke request.
        """
        return {
            "hash_prefix": key_hash[:12],
            "id": record.get("id"),
            "owner_email": record.get("owner_email", ""),
            "scopes": list(record.get("scopes", []) or []),
            "priority_weight": record.get("priority_weight", 1.0),
            "enabled": record.get("enabled", True),
            "created": record.get("created", ""),
            "note": record.get("note", ""),
        }

    def _require_key_issuer(request: Request) -> None:
        principal = getattr(request.state, "principal", None)
        if "key_issuer" not in list(getattr(principal, "scopes", []) or []):
            raise HTTPException(403, detail="key issuer scope required")

    @app.post("/portal/owner-keys/list")
    async def portal_owner_keys_list(request: Request) -> dict[str, Any]:
        """List one verified owner's key metadata for the portal service.

        The relay authenticates the service with ``key_issuer``. The portal is
        responsible for deriving ``owner_email`` from its Okta identity, never
        accepting it from browser input.
        """
        _require_key_issuer(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="invalid JSON")
        email = _owner_email(body)
        records = load_key_records(_keys_path)
        keys = [
            _owner_key_view(key_hash, record)
            for key_hash, record in sorted(records.items())
            if str(record.get("owner_email") or "").strip().lower() == email
        ]
        return {"keys": keys}

    def _owner_principal(records: dict[str, dict[str, Any]], email: str) -> str:
        ids = {
            str(record.get("id") or "").strip()
            for record in records.values()
            if str(record.get("owner_email") or "").strip().lower() == email
        }
        ids.discard("")
        if not ids:
            # New user: derive principal from email slug (same logic as the portal's
            # slug_from_email). This lets first-time CIQ users self-provision.
            import re
            local = email.split("@", 1)[0].strip().lower()
            local = local.split("+", 1)[0].replace(".", "")
            slug = re.sub(r"[^a-z0-9-]+", "-", local).strip("-")[:32]
            if not slug:
                raise HTTPException(400, detail="cannot derive principal from email")
            return slug
        if len(ids) != 1:
            raise HTTPException(409, detail="owner has conflicting principals")
        return ids.pop()

    @app.post("/portal/owner-keys")
    async def portal_owner_keys_add(request: Request) -> dict[str, Any]:
        """Mint one non-admin default token for a mapped portal user."""
        _require_key_issuer(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="invalid JSON")
        email = _owner_email(body)
        principal = _owner_principal(load_key_records(_keys_path), email)
        scopes = ["cloud", "third_party"]
        plaintext = add_key_record(
            _keys_path, principal, priority_weight=1.0, scopes=scopes,
            note="self-provisioned", owner_email=email,
        )
        _reload_principals()
        _audit("owner_key_minted", principal=principal, owner_email=email, by=_acting(request))
        return {"id": principal, "key": plaintext, "scopes": scopes}

    @app.delete("/portal/owner-keys/{hash_prefix}")
    async def portal_owner_keys_revoke(hash_prefix: str, request: Request) -> dict[str, Any]:
        """Revoke one mapped user's non-admin key, never an operator key."""
        _require_key_issuer(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="invalid JSON")
        email = _owner_email(body)
        records = load_key_records(_keys_path)
        matches = [key_hash for key_hash in records if hash_prefix and key_hash.startswith(hash_prefix)]
        if not matches:
            raise HTTPException(404, detail="no owner key matches that prefix")
        if len(matches) != 1:
            raise HTTPException(409, detail="prefix ambiguous; use a longer one")
        record = records[matches[0]]
        if str(record.get("owner_email") or "").strip().lower() != email:
            raise HTTPException(404, detail="no owner key matches that prefix")
        if "admin" in list(record.get("scopes", []) or []):
            raise HTTPException(403, detail="admin key cannot be self-revoked")
        n = revoke_hash(_keys_path, hash_prefix)
        if n != 1:  # Defensive, the exact-match check above makes this unreachable.
            raise HTTPException(409, detail="key changed before revoke")
        _reload_principals()
        _audit("owner_key_revoked", hash_prefix=hash_prefix, owner_email=email, by=_acting(request))
        return {"revoked": 1}

    @app.get("/admin/keys")
    async def admin_keys_list(request: Request) -> dict[str, Any]:
        records = load_key_records(_keys_path)
        return {"keys": [_owner_key_view(key_hash, record)
                         for key_hash, record in sorted(records.items())]}

    @app.post("/admin/keys")
    async def admin_keys_add(request: Request) -> dict[str, Any]:
        body = await request.json()
        if body.get("scopes"):
            raise HTTPException(400, detail="scoped keys are mintable only via the on-box CLI")
        kid = str(body.get("id") or "").strip()
        if not kid:
            raise HTTPException(400, detail="id required")
        owner_email = _owner_email(body) if body.get("owner_email") else None
        scopes = ["cloud", "third_party"]
        plaintext = add_key_record(
            _keys_path, kid,
            priority_weight=float(body.get("priority_weight", 0.5)),
            scopes=scopes,
            note=str(body.get("note", "")),
            owner_email=owner_email,
        )
        _reload_principals()
        _audit("key_minted", principal=kid, by=_acting(request))
        return {"id": kid, "key": plaintext, "scopes": scopes}

    @app.delete("/admin/keys/{hash_prefix}")
    async def admin_keys_revoke(hash_prefix: str, request: Request) -> dict[str, Any]:
        n = revoke_hash(_keys_path, hash_prefix)
        if n == 0:
            raise HTTPException(404, detail="no key matches that prefix")
        if n == -1:
            raise HTTPException(409, detail="prefix ambiguous; use a longer one")
        _reload_principals()
        _audit("key_revoked", hash_prefix=hash_prefix, by=_acting(request))
        return {"revoked": n}

    @app.patch("/admin/keys/{hash_prefix}")
    async def admin_keys_scopes(hash_prefix: str, request: Request) -> dict[str, Any]:
        """Replace the scopes on an existing key. Mirrors the mint restriction:
        the ``admin`` scope is not grantable over HTTP, so a leaked admin key
        can manage users but cannot quietly create more admins."""
        body = await request.json()
        scopes = list(body.get("scopes") or [])
        if "admin" in scopes:
            raise HTTPException(400, detail="admin scope is grantable only via the on-box CLI")
        n = update_key_scopes(_keys_path, hash_prefix, scopes)
        if n == 0:
            raise HTTPException(404, detail="no key matches that prefix")
        if n == -1:
            raise HTTPException(409, detail="prefix ambiguous; use a longer one")
        _reload_principals()
        _audit("key_scopes", hash_prefix=hash_prefix, scopes=scopes, by=_acting(request))
        return {"hash_prefix": hash_prefix, "scopes": scopes}

    # --- Usage read API (admin scope enforced by the middleware). -------------
    # One aggregation path serves every downstream consumer the portal has, so
    # the numbers on the admin cost tab, a user's own usage page, and the WBR
    # collector cannot disagree the way four separate Prometheus queries did.
    @app.get("/admin/usage/rollup")
    async def admin_usage_rollup(request: Request, start: str = "", end: str = "") -> dict[str, Any]:
        """Per (day, principal, client, model) token totals in a date window."""
        from ..usage_query import rollup, valid_day
        from ..usage_store import get_store, open_db

        if not (valid_day(start) and valid_day(end)):
            raise HTTPException(400, detail="start and end must be YYYY-MM-DD")
        store = get_store()
        if store is None:
            return {"rows": [], "enabled": False}
        conn = open_db(store.path)
        try:
            return {"rows": rollup(conn, start, end), "enabled": True}
        finally:
            conn.close()

    @app.get("/admin/usage/latency")
    async def admin_usage_latency(request: Request, start: str = "", end: str = "") -> dict[str, Any]:
        """Per-day exact latency and TTFT percentiles over real observations.

        Exact nearest-rank values over the durations themselves, so the number
        is one some request actually took -- strictly better than the bucketed
        ``histogram_quantile`` estimate the WBR read before. Synthetic backfill
        rows are excluded, so a wholly backfilled day is absent from ``rows``
        rather than reported as zero: the caller must read that as "no samples"
        and keep whatever it already had.
        """
        from ..usage_query import latency, valid_day
        from ..usage_store import get_store, open_db

        if not (valid_day(start) and valid_day(end)):
            raise HTTPException(400, detail="start and end must be YYYY-MM-DD")
        store = get_store()
        if store is None:
            return {"rows": [], "enabled": False}
        conn = open_db(store.path)
        try:
            return {"rows": latency(conn, start, end), "enabled": True}
        finally:
            conn.close()

    @app.get("/admin/usage/summary")
    async def admin_usage_summary(request: Request) -> dict[str, Any]:
        """All-time per-principal totals plus true first/last activity."""
        from ..usage_query import summary
        from ..usage_store import get_store, open_db

        store = get_store()
        if store is None:
            return {"by_principal": {}, "enabled": False}
        conn = open_db(store.path)
        try:
            return {**summary(conn), "enabled": True}
        finally:
            conn.close()

    @app.get("/admin/usage/health")
    async def admin_usage_health(request: Request) -> dict[str, Any]:
        """Store growth and shed-load count: a silently dropping writer is the
        one failure this store exists to make visible."""
        from ..usage_query import store_health
        from ..usage_store import get_store, open_db

        store = get_store()
        if store is None:
            return {"enabled": False, "rows": 0, "bytes": 0, "days": 0}
        conn = open_db(store.path)
        try:
            return {**store_health(conn, store.path), "enabled": True,
                    "dropped": store.dropped}
        finally:
            conn.close()

    @app.get("/admin/usage/cache")
    async def admin_usage_cache(request: Request, start: str = "", end: str = "") -> dict[str, Any]:
        """Per (day, model) prefix-cache reuse, plus a per-model window summary.

        Token-weighted, and in the same convention as every other number this
        API returns: ``input_tokens`` INCLUDES what came from cache, and
        ``cache_read_tokens`` is the of-which subset that avoided prefill --
        OpenAI/OpenRouter's shape, not Anthropic's additive line items. A
        consumer that adds cache reads to input tokens double-counts.

        ``reported`` false means the backend exposes no ``vllm:prefix_cache_*``
        series at all, so reuse for that model is UNKNOWN and ``hit_rate`` is
        None. A reported model that genuinely reused nothing gets a real 0.0.
        Both states exist in production on the same day and neither may
        collapse into the other -- conflating them is the bug class this whole
        lane exists to end.

        The cache tables belong to ``cache_sampler``, not to
        ``usage_store.open_db``, so a store the sampler has never touched has no
        ``cache_daily``. That answers ``sampled: false`` with empty results
        rather than raising ``no such table`` -- the state of every fresh
        deployment, and of every day before sampling began.
        """
        from ..cache_sampler import cache_by_model, cache_rollup, tables_exist
        from ..usage_query import valid_day
        from ..usage_store import get_store, open_db

        if not (valid_day(start) and valid_day(end)):
            raise HTTPException(400, detail="start and end must be YYYY-MM-DD")
        store = get_store()
        if store is None:
            return {"rows": [], "by_model": [], "enabled": False}
        conn = open_db(store.path)
        try:
            if not tables_exist(conn):
                return {"rows": [], "by_model": [], "enabled": True,
                        "sampled": False}
            return {"rows": cache_rollup(conn, start, end),
                    "by_model": cache_by_model(conn, start, end),
                    "enabled": True, "sampled": True}
        finally:
            conn.close()

    # --- Prompt library (admin scope AND a presented admin key). --------------
    # The most sensitive surface in the relay: a searchable archive of
    # coworkers' conversations, retained indefinitely. Authorization is layered
    # rather than assumed from any one check, and every interaction is audited.
    def _require_real_admin_key(request: Request) -> str:
        """Prompt content requires a presented admin key, not the trusted-listener
        bypass.

        That bypass grants admin+cloud+third_party to any request arriving on a
        trusted loopback port with no key at all -- acceptable for operational
        state, not for conversation content. Caddy exposes only the
        key-enforcing listener, so this closes an on-box gap rather than an
        internet-facing one; the portal is unaffected because it already calls
        the auth listener with its service key.

        The scope re-check is deliberate belt-and-braces. The middleware's
        ``_admin_gated`` path prefix already covers ``/admin/*``, but this
        function is the thing a reader of a prompt route will look at, so it
        states its own precondition instead of inheriting it silently.
        """
        source = getattr(request.state, "auth_source", None)
        principal = getattr(request.state, "principal", None)
        caller = getattr(principal, "id", "?")
        scopes = list(getattr(principal, "scopes", []) or [])
        if source != "api_key" or "admin" not in scopes:
            _audit("prompt_access_denied", by=caller, auth_source=str(source),
                   path=request.url.path)
            raise HTTPException(
                403,
                detail="prompt content requires an admin API key "
                       "(the trusted listener is not sufficient)",
            )
        return caller

    @app.get("/admin/prompts/search")
    async def admin_prompts_search(
        request: Request,
        q: str = "",
        principal: str = "",
        client: str = "",
        model: str = "",
        role: str = "",
        start: str = "",
        end: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Full-text search over captured messages. Audited, query text included.

        The audit event fires whether or not capture is configured: switching
        capture off must not silently switch off the record of who went looking.
        """
        from ..prompt_store import get_store, open_db, search

        caller = _require_real_admin_key(request)
        # An unbounded limit on a content route is an exfiltration convenience.
        limit = max(1, min(int(limit), 200))
        store = get_store()
        if store is None:
            _audit("prompt_search", by=caller, query=q, principal=principal,
                   model=model, results=0, enabled=False)
            return {"enabled": False, "query": q, "hits": [], "count": 0,
                    "limit": limit}
        conn = open_db(store.path)
        try:
            # ``search`` sanitises the FTS expression itself, so an unusable
            # query yields no rows rather than a 500 on an admin route.
            hits = search(
                conn, q, principal=principal or None, client=client or None,
                model=model or None, role=role or None,
                start_day=start or None, end_day=end or None, limit=limit,
            )
        finally:
            conn.close()
        _audit("prompt_search", by=caller, query=q, principal=principal,
               model=model, results=len(hits), enabled=True)
        return {"enabled": True, "query": q, "hits": hits, "count": len(hits),
                "limit": limit}

    @app.get("/admin/prompts/request/{request_id}")
    async def admin_prompts_request(request_id: str, request: Request) -> dict[str, Any]:
        """One conversation in full. The audited content read."""
        from ..prompt_store import get_store, open_db, read_request

        caller = _require_real_admin_key(request)
        store = get_store()
        if store is None:
            _audit("prompt_read", by=caller, request_id=request_id,
                   found=False, enabled=False)
            return {"enabled": False, "request_id": request_id,
                    "found": False, "messages": []}
        conn = open_db(store.path)
        try:
            out = read_request(conn, request_id)
        finally:
            conn.close()
        _audit("prompt_read", by=caller, request_id=request_id,
               found=bool(out.get("found")), enabled=True)
        return {**out, "enabled": True}

    @app.get("/admin/prompts/stats")
    async def admin_prompts_stats(request: Request) -> dict[str, Any]:
        """Row counts, dedup ratio and on-disk size. Metadata only, no content.

        Gated and audited like the content routes anyway: growth of an
        indefinitely-retained archive is itself something to watch, and one
        consistent gate is easier to reason about than two.
        """
        from ..prompt_store import get_store, open_db, stats

        caller = _require_real_admin_key(request)
        store = get_store()
        if store is None:
            _audit("prompt_stats", by=caller, enabled=False)
            return {"enabled": False, "stored_messages": 0, "message_links": 0,
                    "requests": 0, "bytes": 0}
        conn = open_db(store.path)
        try:
            out = stats(conn, store.path)
        finally:
            conn.close()
        _audit("prompt_stats", by=caller, enabled=True)
        return {**out, "enabled": True, "dropped": store.dropped}

    @app.get("/routing-table")
    async def routing_table(request: Request) -> dict[str, list[str]]:
        return dict(request.app.state.config.policy.fallback.graph)

    @app.get("/routing-table/{model}")
    async def routing_table_for(model: str, request: Request) -> dict[str, Any]:
        cfg_all = request.app.state.config
        m = cfg_all.models.models.get(model)
        if m:
            return {
                "model": model,
                "provider": m.provider,
                "fallback_chain": request.app.state.router.selector.get_fallback_chain(model),
            }
        # Aliases are first-class here too ("what does this alias resolve to
        # right now"): use-case categories are derived, not static models, and
        # health-gate tooling probes e.g. /routing-table/main.
        members = cfg_all.models.aliases.get(model)
        if members:
            disc = request.app.state.discovery
            resolved = None
            for member in members:
                if member in cfg_all.models.models and disc.get_model_state(member).value in (
                    "available",
                    "degraded",
                ):
                    resolved = member
                    break
            return {"alias": model, "members": list(members), "resolved": resolved}
        raise HTTPException(404, detail=f"Unknown model: {model}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="Invalid JSON")
        # NOTE: this allowlist is the door. A routing header parsed in the router
        # but missing here is silently dropped and the axis ships inert — that is
        # exactly how X-Llm-Relay-Candidate-Lane shipped dead. Add the header here
        # in the same change that parses it, and cover it with a TestClient test.
        hint_headers: dict[str, str] = {}
        for key in (
            "X-Llm-Relay-Privacy",
            "X-Llm-Relay-Confidentiality",
            "X-Llm-Relay-Require-Tools",
            "X-Llm-Relay-Min-Context",
        ):
            v = request.headers.get(key)
            if v is not None:
                hint_headers[key] = v
        _principal = getattr(request.state, "principal", None)
        _clamp_privacy(_principal, request.app.state.config.auth.enabled, hint_headers)
        _clamp_confidentiality(_principal, request.app.state.config.auth.enabled, hint_headers)
        user_agent = request.headers.get("user-agent", "")
        # Explicit X-Llm-Relay-Client header wins; else fall back to a configured
        # distinctive User-Agent pattern; else "unknown". The known-client set and
        # UA patterns are deployment-configured via the environment (see
        # metrics.configure_clients_from_env), so a client with a distinctive UA
        # can be attributed with zero client-side change while others opt in via
        # the header.
        client = resolve_client(request.headers.get("X-Llm-Relay-Client"), user_agent)
        principal_id = getattr(_principal, "id", "anonymous")
        start_ns = time.time_ns()
        is_stream = body.get("stream") is True

        # QoS admission (plan 4, slice 1): shed explicitly low-urgency work under
        # fleet contention so high-urgency / interactive work keeps flowing.
        if request.app.state.admission.should_shed(
            request.headers.get("X-Llm-Relay-Urgency"), request.app.state.discovery
        ):
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=429, streamed=is_stream, error="shed: low urgency under contention",
                outcome="shed", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            raise HTTPException(
                status_code=429,
                detail={"error": "shed under contention (low urgency); retry shortly"},
                headers={"Retry-After": "5"},
            )

        try:
            if is_stream:
                upstream, body_iter, result, cleanup = await request.app.state.router.route_and_forward(
                    request_data=body, headers=hint_headers, stream=True,
                )
            else:
                _fwd_task = asyncio.ensure_future(
                    request.app.state.router.route_and_forward(
                        request_data=body, headers=hint_headers, stream=False,
                    )
                )
                try:
                    upstream, result = await asyncio.wait_for(
                        asyncio.shield(_fwd_task), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Backend needs >60s (ornith-397b etc). Drip whitespace
                    # keepalives so Cloudflare's 100s edge timeout stays open.
                    async def _keepalive_drip():
                        try:
                            while not _fwd_task.done():
                                yield b" "
                                done, _ = await asyncio.wait(
                                    {_fwd_task}, timeout=30
                                )
                                if done:
                                    break
                            _up, _res = _fwd_task.result()
                        except Exception as _exc:
                            _err = {
                                "error": {
                                    "message": f"Backend error: {_exc}",
                                    "type": "relay_error",
                                }
                            }
                            emit_chat_completion(
                                request_body=body,
                                response_body=None,
                                response_text=None,
                                usage=None,
                                model_resolved=None,
                                provider_name=None,
                                user_agent=user_agent,
                                start_ns=start_ns,
                                end_ns=time.time_ns(),
                                status_code=502,
                                streamed=False,
                                error=str(_exc),
                                outcome="backend_error",
                                client=client,
                                principal=principal_id,
                                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
                            )
                            yield json.dumps(_err).encode()
                            return
                        try:
                            _content = _up.json()
                            if isinstance(_content, dict):
                                _content["llm-relay"] = {
                                    "selected_model": _res.selected_model,
                                    "selected_provider": _res.provider_name,
                                    "decision": _res.decision,
                                }
                        except Exception:
                            _content = {"raw": _up.text}
                        emit_chat_completion(
                            request_body=body,
                            response_body=(
                                _content
                                if isinstance(_content, dict)
                                else None
                            ),
                            response_text=(
                                None
                                if isinstance(_content, dict)
                                else _up.text
                            ),
                            usage=None,
                            model_resolved=_res.selected_model,
                            provider_name=_res.provider_name,
                            user_agent=user_agent,
                            start_ns=start_ns,
                            end_ns=time.time_ns(),
                            status_code=_up.status_code,
                            streamed=False,
                            outcome=(
                                "success"
                                if _up.status_code < 400
                                else "upstream_error"
                            ),
                            client=client,
                            principal=principal_id,
                            fell_back=did_fall_back(
                                _res.selected_model,
                                (_res.decision or {}).get("ranked") or [],
                            ),
                            confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
                        )
                        if isinstance(_content, dict):
                            yield json.dumps(_content).encode()
                        else:
                            yield _up.text.encode()

                    return StreamingResponse(
                        _keepalive_drip(),
                        status_code=200,
                        media_type="application/json",
                    )
        except ContextLengthExceededError as e:
            # Prompt exceeds every servable model's window. Return the OpenAI-standard
            # top-level 400 context_length_exceeded so clients auto-compact and retry
            # smaller, instead of treating a 5xx as transient and retry-dying.
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=400, streamed=is_stream, error=str(e),
                outcome="context_length_exceeded", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            return JSONResponse(
                status_code=400, content=e.body,
                headers={"X-Llm-Relay-Error": "context_length_exceeded"},
            )
        except SaturationError as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=429, streamed=is_stream, error=str(e),
                outcome="saturated", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            return _backpressure_response(
                429, "backend_saturated", str(e), e.retry_after_seconds,
                {"backend": e.backend_key},
            )
        except NoBackendAvailableError as e:
            # Transient availability gap (constraints satisfiable, every match
            # momentarily down/paused): 503 + Retry-After backpressure, a distinct
            # outcome from saturation (slots full) and from a genuine no-candidate.
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=503, streamed=is_stream, error=str(e),
                outcome="no_backend", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            return _backpressure_response(
                503, "no_backend_available", str(e), e.retry_after_seconds,
            )
        except httpx.RequestError as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=is_stream, error=f"Backend network error: {e}",
                outcome="network_error", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            raise HTTPException(502, detail=f"Backend network error: {e}")
        except HTTPException as http_exc:
            # route_and_forward raises HTTPException for no-candidates 503.
            # A NAMED-model refusal records as its own reason-coded outcome
            # (observation-first-health-spec §3.4) with the model it actually
            # refused — before this, it was lumped into no_candidate with
            # model=None, which is exactly how a fifth of fleet requests hid
            # inside the WBR's "context rejects" bucket. Reasons are a small
            # closed set, so metric cardinality stays bounded.
            named_model = None
            named_reason = ""
            named_provider = None
            if isinstance(http_exc.detail, dict):
                nm = http_exc.detail.get("named_model")
                if isinstance(nm, dict):
                    named_model = nm.get("model")
                    named_provider = nm.get("provider")
                    avail = nm.get("availability")
                    if isinstance(avail, dict) and avail.get("reason"):
                        named_reason = str(avail["reason"])
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=named_model, provider_name=named_provider,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=503, streamed=is_stream, error="No model matches constraints",
                outcome=(f"named_unavailable_{named_reason or 'unknown'}"
                         if named_model else "no_candidate"),
                client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            raise
        except Exception as e:
            emit_chat_completion(
                request_body=body, response_body=None, response_text=None, usage=None,
                model_resolved=None, provider_name=None,
                user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                status_code=502, streamed=is_stream, error=f"Backend error: {e}",
                outcome="backend_error", client=client, principal=principal_id,
                confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
            )
            raise HTTPException(502, detail=f"Backend error: {e}")

        _dec = result.decision or {}
        # reasoning->reasoning_content mirroring is applied ONLY for models that
        # advertise the `reasoning` capability. For every other backend the
        # response is passed through byte-identical (no parse, no re-serialize),
        # so this is zero-overhead and zero-risk for the fast local fleet.
        _sel_cfg = request.app.state.config.models.models.get(result.selected_model or "")
        _mirror_r = bool(_sel_cfg and "reasoning" in (_sel_cfg.capabilities or []))
        relay_headers = {
            "X-Llm-Relay-Selected-Model": result.selected_model or "",
            "X-Llm-Relay-Selected-Provider": result.provider_name or "",
            # 4-tuple decision (plan 3): quant + node + batch policy. quant is the
            # highest-preference variant's precision (a side effect of quality
            # ordering, not an independent cost axis). Set on BOTH response paths.
            "X-Llm-Relay-Decision-Quant": str(_dec.get("quant") or ""),
            "X-Llm-Relay-Decision-Node": str(_dec.get("node") or ""),
            "X-Llm-Relay-Decision-Batch": str(_dec.get("batch") or ""),
        }

        if is_stream:
            media_type = upstream.headers.get("content-type", "text/event-stream")
            upstream_status = upstream.status_code

            async def _tee_and_emit():
                chunks: list[bytes] = []
                exc: BaseException | None = None
                first_chunk_ns: int | None = None
                carry = b""  # partial trailing SSE frame awaiting its blank-line terminator
                # In-band degeneracy detector: check the accumulated content
                # every ~64 tokens for repetition loops. If detected, abort
                # the stream with a terminal error event instead of piping
                # thousands of repeating tokens to the client.
                _deg_check_text = ""
                _deg_check_count = 0
                _deg_aborted = False
                try:
                    async for chunk, _is_ka in _sse_stream_keepalive(
                        body_iter, media_type, _KEEPALIVE_INTERVAL_S
                    ):
                        if _is_ka:
                            yield chunk
                            continue
                        if first_chunk_ns is None:
                            first_chunk_ns = time.time_ns()
                        if not _mirror_r:
                            # Non-reasoning model: byte-identical passthrough.
                            chunks.append(chunk)
                            yield chunk
                            # Degeneracy check: extract content from SSE chunks
                            # and check every ~256 chars (~64 tokens)
                            try:
                                for line in chunk.decode("utf-8", "replace").split("\n"):
                                    if line.startswith("data:") and "[DONE]" not in line:
                                        d = json.loads(line[5:].strip())
                                        delta = (d.get("choices", [{}])[0].get("delta") or {})
                                        c = delta.get("content", "") or ""
                                        if c:
                                            _deg_check_text += c
                                            _deg_check_count += 1
                                            if _deg_check_count >= 64 and len(_deg_check_text) > 256:
                                                if is_degenerate(_deg_check_text):
                                                    _deg_aborted = True
                                                    err = {"error": {"message": "generation degenerate (repetition loop detected); aborting", "type": "degenerate"}}
                                                    yield b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n"
                                                    return
                                                _deg_check_count = 0
                            except Exception:
                                pass  # don't let degeneracy checking crash the stream
                            continue
                        # Reasoning model: mirror reasoning->reasoning_content on
                        # COMPLETE frames only (split on the blank-line terminator),
                        # buffering any partial tail so a JSON payload is never split
                        # mid-frame or mid-codepoint. Keepalives are handled above and
                        # never enter this buffer.
                        carry += chunk
                        while b"\n\n" in carry:
                            fb, carry = carry.split(b"\n\n", 1)
                            out = (_mirror_reasoning_sse_frame(fb.decode("utf-8", "replace"))
                                   + "\n\n").encode("utf-8")
                            chunks.append(out)
                            yield out
                    if _mirror_r and carry:
                        # Flush a trailing frame that lacked its terminator (rare).
                        out = _mirror_reasoning_sse_frame(carry.decode("utf-8", "replace")).encode("utf-8")
                        chunks.append(out)
                        yield out
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
                    if _deg_aborted:
                        outcome = "degenerate"
                    # End-to-end TTFT (first chunk ~= first byte, includes routing);
                    # None when no chunk ever flowed.
                    ttft_ns = (first_chunk_ns - start_ns) if first_chunk_ns is not None else None
                    emit_chat_completion(
                        request_body=body, response_body=None, response_text=text, usage=usage,
                        model_resolved=result.selected_model, provider_name=result.provider_name,
                        user_agent=user_agent, start_ns=start_ns, end_ns=time.time_ns(),
                        status_code=upstream_status, streamed=True,
                        outcome=outcome,
                        client=client, principal=principal_id,
                        fell_back=did_fall_back(result.selected_model, (result.decision or {}).get("ranked") or []),
                        ttft_ns=ttft_ns,
                        confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
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
                if _mirror_r:
                    _mirror_reasoning(content)
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
            client=client, principal=principal_id,
            fell_back=did_fall_back(result.selected_model, (result.decision or {}).get("ranked") or []),
            confidentiality=hint_headers.get("X-Llm-Relay-Confidentiality"),
        )
        return JSONResponse(status_code=upstream.status_code, content=content, headers=relay_headers)

    async def _simple_proxy(request: Request, upstream_path: str):
        """Shared handler for simple non-streaming endpoints (embeddings, rerank).
        Routes by the requested model and forwards to ``upstream_path`` (plan 6)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="Invalid JSON")
        hint_headers = {
            k: request.headers[k]
            for k in ("X-Llm-Relay-Privacy", "X-Llm-Relay-Confidentiality")
            if k in request.headers
        }
        _clamp_privacy(
            getattr(request.state, "principal", None),
            request.app.state.config.auth.enabled,
            hint_headers,
        )
        _clamp_confidentiality(
            getattr(request.state, "principal", None),
            request.app.state.config.auth.enabled,
            hint_headers,
        )
        try:
            upstream, result = await request.app.state.router.route_simple(
                body, headers=hint_headers, upstream_path=upstream_path,
            )
        except SaturationError as e:
            return _backpressure_response(
                429, "backend_saturated", str(e), e.retry_after_seconds,
            )
        except NoBackendAvailableError as e:
            return _backpressure_response(
                503, "no_backend_available", str(e), e.retry_after_seconds,
            )
        except httpx.RequestError as e:
            raise HTTPException(502, detail=f"Backend network error: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, detail=f"Backend error: {e}")
        try:
            content = upstream.json()
        except Exception:
            content = {"raw": upstream.text}
        return JSONResponse(
            status_code=upstream.status_code, content=content,
            headers={"X-Llm-Relay-Selected-Model": result.selected_model or ""},
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _simple_proxy(request, "embeddings")

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        return await _simple_proxy(request, "rerank")

    @app.get("/logs")
    async def logs(request: Request) -> Response:
        """Recent buffered relay log lines (plain text), for the cockpit (plan 7)."""
        buf = request.app.state.log_buffer
        return Response(content="\n".join(buf.recent(limit=500)), media_type="text/plain")

    @app.get("/logs/stream")
    async def logs_stream(request: Request) -> StreamingResponse:
        """SSE of relay log lines: recent history first, then new lines as they
        arrive (poll-based over a monotonic sequence). Stops on client disconnect."""
        buf = request.app.state.log_buffer

        async def gen():
            last = 0
            for s, line in buf.since(0)[-200:]:
                yield f"data: {line}\n\n"
                last = max(last, s)
            while True:
                if await request.is_disconnected():
                    break
                for s, line in buf.since(last):
                    yield f"data: {line}\n\n"
                    last = s
                await asyncio.sleep(1.0)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.post("/v1/jobs")
    async def submit_job(request: Request):
        """Submit an agentic chat job to the async lane (plan 4 slice 2). Returns a
        job id; poll GET /v1/jobs/{id} for status and result. The job survives a
        relay restart (durable store)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, detail="Invalid JSON")
        principal = getattr(request.state, "principal", None)
        job = request.app.state.job_store.create(
            principal=getattr(principal, "id", "anonymous"),
            body=body,
            sla_class=request.headers.get("X-Llm-Relay-SLA-Class"),
            urgency=request.headers.get("X-Llm-Relay-Urgency"),
            priority_weight=getattr(principal, "priority_weight", 1.0),
            created_ts=time.time(),
        )
        return JSONResponse(status_code=202, content={"job_id": job.id, "status": job.status})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        job = request.app.state.job_store.get(job_id)
        if job is None or not _job_visible(
            job, getattr(request.state, "principal", None),
            request.app.state.config.auth.enabled,
        ):
            # 404 (not 403) for another principal's job: don't leak existence.
            raise HTTPException(404, detail=f"Unknown job: {job_id}")
        return job.public()

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        store = request.app.state.job_store
        job = store.get(job_id)
        if job is None or not _job_visible(
            job, getattr(request.state, "principal", None),
            request.app.state.config.auth.enabled,
        ):
            raise HTTPException(404, detail=f"Unknown job: {job_id}")
        cancelled = store.cancel(job_id)
        return {"job_id": job_id, "cancelled": cancelled, "status": store.get(job_id).status}

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


def build_sockets(host: str, port: int, auth_port: int | None, auth_cfg) -> tuple[list, list[str]]:
    """Listener sockets for :func:`serve`. Fail-closed: the auth listener is
    refused (with a warning) when auth is enabled but the key store has no
    enabled key, so a misconfigured deployment cannot expose a keyless "auth"
    port. The trusted/primary listener always binds so local consumers stay up.
    """
    import socket as _socket

    def _bind(p: int):
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        s.bind((host, p))
        s.listen(2048)
        return s

    sockets, warnings = [_bind(port)], []
    if auth_port is not None:
        has_key = any(p.enabled for p in auth_cfg.principals_by_hash.values())
        if auth_cfg.enabled and not has_key:
            warnings.append(
                f"auth listener :{auth_port} REFUSED: auth enabled but api_keys.yaml has no enabled key"
            )
        else:
            sockets.append(_bind(auth_port))
    return sockets, warnings


async def serve(config_dir: str | Path | None = None) -> None:
    """Run one relay process on one or two listeners (single lifespan).

    ``LLM_RELAY_PORT`` is the primary listener (typically a trusted loopback
    port, see auth.trusted_ports); ``LLM_RELAY_AUTH_PORT``, when set, adds the
    key-enforced listener a reverse proxy routes external traffic to.
    """
    import logging

    log = logging.getLogger("llm_relay")
    host = os.environ.get("LLM_RELAY_HOST", "127.0.0.1")
    port = int(os.environ.get("LLM_RELAY_PORT", 8090))
    auth_port_raw = os.environ.get("LLM_RELAY_AUTH_PORT", "")
    auth_port = int(auth_port_raw) if auth_port_raw else None
    app = create_app(config_dir)
    sockets, warnings = build_sockets(host, port, auth_port, app.state.config.auth)
    for w in warnings:
        log.error(w)
    log.info(
        "listeners: %s (primary=%s auth=%s trusted_ports=%s)",
        [s.getsockname()[1] for s in sockets], port, auth_port,
        app.state.config.auth.trusted_ports,
    )
    config = uvicorn.Config(app, log_level=os.environ.get("LLM_RELAY_LOG_LEVEL", "info"))
    server = uvicorn.Server(config)
    await server.serve(sockets=sockets)


if __name__ == "__main__":
    asyncio.run(serve())
