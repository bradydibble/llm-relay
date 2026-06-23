"""Request routing and upstream forwarding."""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import HTTPException

from ..config.loader import ConfigLoader
from ..config.types import NoBackendAvailableError, Privacy, SaturationError
from ..discovery.endpoint import _shared_upstream_bearer
from ..discovery.manager import DiscoveryManager
from .keys import compose_backend_key, compose_backend_url
from .selector import ChainCandidate, ModelSelector, RoutingContext, batch_policy_for


# A model is eligible for a request when it can hold the PROMPT plus this much
# output headroom. The client's max_tokens (an output ceiling) is NOT added to
# the eligibility floor — it is clamped to the chosen model's headroom at forward
# time (see _clamp_max_tokens). So a generous max_tokens neither widens nor pins
# routing; only a prompt that genuinely fits nothing live is refused (oversize).
MIN_OUTPUT_HEADROOM = 1024

# Retry-After hint (seconds) for a TRANSIENT no-candidate: the constraints are
# satisfiable but every matching backend is momentarily down/paused. Sized to the
# ~15s discovery poll cadence, so a recovered/unpaused backend is re-detected
# within roughly one retry. The caller backs off and retries instead of treating
# the empty chain as terminal.
NO_BACKEND_RETRY_AFTER = 15.0


@dataclass
class RouteResult:
    success: bool
    selected_model: str | None
    backend_url: str | None
    provider_name: str | None
    error: str | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    backend_key: str | None = None
    slot_wait_timeout: float = 30.0


class RequestRouter:
    def __init__(self, config: ConfigLoader, discovery: DiscoveryManager):
        self.config = config
        self.discovery = discovery
        self.selector = ModelSelector(config, discovery)

    def _backend_url(self, model_name: str) -> str | None:
        cfg = self.config.models.models.get(model_name)
        if not cfg:
            return None
        provider = self.config.providers.get(cfg.provider)
        if not provider:
            return None
        return compose_backend_url(provider.base_url, cfg.port, cfg.path)

    def _apply_filters(self, body: dict[str, Any], model_name: str) -> dict[str, Any]:
        """Strip/override request params per the model's configured filters
        (plan 5), before the request hits the upstream. Returns a new dict and
        never mutates the caller's body; returns the same object when there is
        nothing to do. Applied before ``model`` is set, so a filter cannot drop
        or rewrite the routed model id."""
        # None-safe: stream_request is exercised with config=None in tests (its
        # slot/stream logic must not depend on config), so a missing config simply
        # means no filters.
        config = self.config
        if config is None:
            return body
        cfg = config.models.models.get(model_name)
        if not cfg:
            return body
        strip = cfg.strip_params or []
        setp = cfg.set_params or {}
        if not strip and not setp:
            return body
        out = {k: v for k, v in body.items() if k not in strip}
        out.update(setp)
        return out

    async def forward_request(
        self,
        backend_url: str,
        model_name: str,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        backend_key: str | None = None,
        slot_wait_timeout: float = 30.0,
        upstream_path: str = "chat/completions",
    ) -> httpx.Response:
        body = self._apply_filters(dict(request_data), model_name)
        body["model"] = model_name
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        # Authenticate to api-key'd upstreams using the shared homelab bearer
        # (see endpoint.py for env-var resolution). Caller-provided
        # Authorization wins so a future per-request auth path can override.
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # Structured timeout, mirroring the streaming path (see stream_request): a
        # GENEROUS read window so a slow large completion runs to completion on the
        # local 35B (a ~70k prompt prefills 100-250s+ before the first byte), but a
        # SHORT connect so a genuinely dead backend fails fast instead of holding the
        # slot for the whole window. The old flat 300s TOTAL cap silently overrode a
        # caller's longer client timeout (the wiki engine sets 900s) and killed any
        # non-stream completion past five minutes — an arbitrary cutoff on hardware
        # that is idle most of the day. The read window matches the engine's 900s.
        timeout = httpx.Timeout(connect=10.0, read=900.0, write=10.0, pool=10.0)
        # backend_key="" / None → acquire_slot is a no-op (no semaphore registered).
        async with self.discovery.acquire_slot(backend_key or "", wait_timeout=slot_wait_timeout):
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(
                    f"{backend_url}/{upstream_path}",
                    json=body,
                    headers=merged_headers,
                )

    async def stream_request(
        self,
        backend_url: str,
        model_name: str,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        backend_key: str | None = None,
        slot_wait_timeout: float = 30.0,
    ) -> tuple[httpx.Response, AsyncIterator[bytes], Callable[[], Awaitable[None]]]:
        """Open a streaming upstream connection.

        Returns ``(response, body_iterator, cleanup)``. The caller reads
        status/headers off ``response``, streams ``body_iterator``, and MUST
        ensure ``cleanup`` runs once the response is finished — wire it as the
        ``StreamingResponse`` background task. ``cleanup`` is idempotent: the
        iterator's ``finally`` also invokes it, so whichever fires first wins
        and the slot is freed promptly without waiting on generator GC.

        The in-flight slot is held from acquire (here) until the iterator is
        exhausted or aborted. We acquire a :class:`SlotHandle` rather than an
        ``async with`` block because the slot lifetime must span the returned
        generator, and the release must be a *synchronous* call: a client
        disconnect cancels the generator, and a release sitting behind an
        ``await`` can be preempted by that cancellation — which is exactly how
        the slot used to leak.
        """
        body = self._apply_filters(dict(request_data), model_name)
        body["model"] = model_name
        # Ask the upstream to include token usage in the final SSE event. Standard
        # OpenAI streaming omits usage; this opts back in so cross-provider captures
        # (Anthropic fallback, etc.) have token counts without relying on
        # llama-server's non-standard `timings` field. Preserve any user override.
        existing_opts = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        body["stream_options"] = {"include_usage": True, **existing_opts}
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # 600s per-chunk read timeout — SSE may legitimately stall between tokens
        # on slow models within that window, but a truly dead upstream gets canceled.
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)

        # Acquire the slot BEFORE building the client so a SaturationError never
        # leaks an open connection. The handle's release is synchronous and runs
        # FIRST in the iterator's finally — see the SlotHandle docstring.
        handle = await self.discovery.acquire_slot_handle(
            backend_key or "", wait_timeout=slot_wait_timeout,
        )  # may raise SaturationError; propagates to caller

        client = httpx.AsyncClient(timeout=timeout)
        try:
            req = client.build_request(
                "POST",
                f"{backend_url}/chat/completions",
                json=body,
                headers=merged_headers,
            )
            resp = await client.send(req, stream=True)
        except BaseException:
            handle.release()
            await client.aclose()
            raise

        cleaned = False

        async def _cleanup() -> None:
            """Idempotent per-request teardown: free the slot, then close the
            response and client.

            Wired in two places — the iterator's ``finally`` and the
            StreamingResponse background task — so cleanup survives whichever
            path FastAPI takes (normal drain, upstream error, or client
            disconnect). Slot release runs first and synchronously on every
            call, so a CancelledError on the later ``await``s can never strand
            the slot; connection teardown runs once.
            """
            nonlocal cleaned
            handle.release()
            if cleaned:
                return
            cleaned = True
            with contextlib.suppress(Exception):
                await resp.aclose()
            with contextlib.suppress(Exception):
                await client.aclose()

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await _cleanup()

        return resp, _iter(), _cleanup

    async def route_and_forward(
        self,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        stream: bool = False,
    ):
        """Resolve the fallback chain and forward, retrying on retry_on errors.

        Non-streaming returns ``(httpx.Response, RouteResult)``.
        Streaming returns ``(httpx.Response, AsyncIterator[bytes], RouteResult,
        cleanup)`` — the API layer wires ``cleanup`` as the response background
        task. Streaming does NOT retry across candidates (see note below).

        Behavior
        --------
        - Walks the candidate chain in priority order.
        - Non-streaming: on a retry_on HTTP status (default 502/503/504) or a
          retry_on network exception (ConnectError, ReadTimeout,
          RemoteProtocolError), tries the next candidate.
        - Streaming: routes once with the existing ``stream_request`` path —
          no cross-backend retry. Retry is deferred because a streamed response
          can't be replayed across backends once bytes have flowed. (The slot
          is no longer at risk on abort: ``stream_request`` releases it
          synchronously and the API wires ``cleanup`` as a background task.)
        - ``SaturationError`` propagates IMMEDIATELY — slot saturation is
          backpressure, not a broken backend.  The caller should back off via
          ``Retry-After``, not amplify load by trying other backends.
        - Non-retryable upstream statuses (e.g. 400, 401) propagate as-is.
        - If the chain is exhausted, the last observed error or response is
          surfaced.
        """
        headers = headers or {}
        privacy_str = headers.get("X-Llm-Relay-Privacy", "local_only")
        privacy = Privacy(privacy_str if privacy_str in ("local_only", "cloud_ok") else "local_only")

        # Context-aware routing: an explicit X-Llm-Relay-Min-Context header is a
        # floor the caller asserts; we also size the request from its PROMPT and
        # take the larger. The selector drops candidates whose window is below this
        # floor, so the prompt is never routed to a backend too small to hold it.
        # max_tokens is NOT in the floor (adding it pins every generous request to
        # the single largest backend); the output ceiling is fitted per-candidate
        # by _clamp_max_tokens at forward time.
        explicit_min = int(headers.get("X-Llm-Relay-Min-Context", "0") or 0)
        prompt_est = _estimate_prompt_tokens(request_data)
        estimated_min = (prompt_est + MIN_OUTPUT_HEADROOM) if prompt_est else 0
        # Client-declared intent (plan 3): SLA class + urgency (recorded, used by
        # the scheduler in plan 4) and an optional quality floor (parsed into
        # min_preference; combined with any category floor downstream).
        sla_class = headers.get("X-Llm-Relay-SLA-Class") or None
        urgency = headers.get("X-Llm-Relay-Urgency") or None
        quality_floor: float | None = None
        qf_raw = headers.get("X-Llm-Relay-Quality-Floor")
        if qf_raw:
            try:
                quality_floor = float(qf_raw)
            except ValueError:
                quality_floor = None
        ctx = RoutingContext(
            requested_model=request_data.get("model", "") or "",
            privacy=privacy,
            require_tools=headers.get("X-Llm-Relay-Require-Tools", "false").lower() == "true",
            min_context=max(explicit_min, estimated_min) or None,
            min_preference=quality_floor,
            sla_class=sla_class,
            urgency=urgency,
        )

        candidates = self.selector.select_chain(ctx)
        if not candidates:
            detail = {
                "error": "No model matches constraints",
                "decision": {
                    "requested": ctx.requested_model,
                    "candidates": ctx.candidates,
                    "filtered": ctx.filtered,
                },
            }
            # When the binding constraint is context (the request can't fit any live
            # model), attach an actionable signal: oversize_for_now (wait for a
            # big-enough model to return) vs oversize_period (resize / defer). The
            # client adapts deterministically; the relay never silently truncates.
            shortfall = self.selector.diagnose_context_shortfall(ctx)
            if shortfall is not None:
                detail["context"] = shortfall
                raise HTTPException(status_code=503, detail=detail)
            # Not a context shortfall: if the constraints WOULD be met by a
            # configured model that's merely down/paused right now (a discovery
            # blip or a maintenance pause), the empty chain is a TRANSIENT
            # availability gap — answer with Retry-After backpressure so batch
            # callers wait and retry, instead of a terminal "No model matches
            # constraints". A genuine mismatch (nothing can ever match) stays terminal.
            if self.selector.is_transient_no_candidate(ctx):
                raise NoBackendAvailableError(retry_after_seconds=NO_BACKEND_RETRY_AFTER)
            raise HTTPException(status_code=503, detail=detail)

        # "connection_error" in retry_on means network exceptions; HTTP codes
        # are matched as strings against str(resp.status_code).
        retry_codes: set[str] = {
            code for code in self.config.policy.fallback.retry_on
            if code != "connection_error"
        }
        retry_exceptions = (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        )

        # Streaming: spill past a saturated candidate BEFORE the first byte.
        # Saturation is decided at slot-acquire, before any SSE byte flows, so
        # choosing another backend here is pre-flight-safe — unlike mid-stream
        # failover, which is unreplayable (see docstring).
        if stream:
            saturation_error: SaturationError | None = None
            last_error: Exception | None = None
            # A retry-status candidate held pending a better one. Pre-first-byte:
            # we have the upstream status but have read no SSE body, so we can
            # still abandon it for the next candidate. Only ONE is held at a time
            # (the prior is freed before a newer is kept), so at most one extra
            # slot is occupied transiently — and it is always released, never leaked.
            pending: tuple | None = None
            for candidate in candidates:
                # Skip a backend with no free slot WITHOUT waiting on it — same
                # pre-flight spill as the non-streaming path.
                if not self.discovery.has_free_slot(candidate.backend_key):
                    if saturation_error is None:
                        saturation_error = SaturationError(
                            backend_key=candidate.backend_key,
                            retry_after_seconds=candidate.slot_wait_timeout,
                        )
                    continue
                try:
                    fwd = _clamp_max_tokens(request_data, prompt_est, candidate.context_window)
                    upstream, body_iter, cleanup = await self.stream_request(
                        candidate.backend_url, candidate.model, fwd,
                        headers=headers,
                        backend_key=candidate.backend_key,
                        slot_wait_timeout=candidate.slot_wait_timeout,
                    )
                except SaturationError as exc:
                    saturation_error = saturation_error or exc
                    continue
                except retry_exceptions as exc:
                    # Connect-phase failure, before any byte — stream_request has
                    # already released its own slot, so just try the next candidate.
                    last_error = exc
                    continue
                route_result = _candidate_to_route_result(candidate, ctx)
                if str(upstream.status_code) in retry_codes:
                    # Retryable upstream status, still pre-first-byte: free any
                    # prior pending stream and hold this one while we try the rest.
                    if pending is not None:
                        await pending[3]()
                    pending = (upstream, body_iter, route_result, cleanup)
                    continue
                # Success — or a non-retryable status (e.g. 400) we must not burn
                # the chain on. Commit to it; free any pending retry-status stream.
                if pending is not None:
                    await pending[3]()
                return upstream, body_iter, route_result, cleanup
            # Chain exhausted.
            if pending is not None:
                # Every candidate gave a retryable status — return the last so the
                # client still sees the upstream 5xx (as the single-candidate path
                # did). Earlier candidates were already cleaned up above.
                return pending
            if saturation_error is not None:
                raise saturation_error
            if last_error is not None:
                raise last_error
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "No model matches constraints",
                    "decision": {
                        "requested": ctx.requested_model,
                        "candidates": ctx.candidates,
                        "filtered": ctx.filtered,
                    },
                },
            )

        # Non-streaming: walk the chain, retry on retry_on errors.
        last_response: httpx.Response | None = None
        last_response_candidate: ChainCandidate | None = None
        last_error: Exception | None = None
        # Set when a candidate is saturated. Saturation is spilled past
        # candidate-by-candidate and only surfaced if the WHOLE chain is full —
        # at which point backpressure (503 + Retry-After) is the correct answer.
        saturation_error: SaturationError | None = None

        for candidate in candidates:
            route_result = _candidate_to_route_result(candidate, ctx)
            # Spill past a backend with no free slot WITHOUT waiting on it: a single
            # saturated backend is not a reason to fail when another can serve the
            # request. Skipping here also avoids paying slot_wait_timeout per
            # already-full candidate before falling through.
            if not self.discovery.has_free_slot(candidate.backend_key):
                if saturation_error is None:
                    saturation_error = SaturationError(
                        backend_key=candidate.backend_key,
                        retry_after_seconds=candidate.slot_wait_timeout,
                    )
                continue
            try:
                fwd = _clamp_max_tokens(request_data, prompt_est, candidate.context_window)
                resp = await self.forward_request(
                    candidate.backend_url, candidate.model, fwd,
                    headers=headers,
                    backend_key=candidate.backend_key,
                    slot_wait_timeout=candidate.slot_wait_timeout,
                )
                if str(resp.status_code) in retry_codes:
                    last_response = resp
                    last_response_candidate = candidate
                    continue
                return resp, route_result
            except SaturationError as exc:
                # Raced: passed the free-slot check but filled before acquire.
                # Treat like any other saturated candidate — spill to the next.
                if saturation_error is None:
                    saturation_error = exc
                continue
            except retry_exceptions as exc:
                last_error = exc
                continue

        # Chain exhausted — surface the last observed error/response.
        # Use last_response_candidate (the one that produced last_response),
        # NOT candidates[-1] (the last *attempted* — which may have network-errored).
        if last_response is not None:
            final_result = _candidate_to_route_result(last_response_candidate, ctx)  # type: ignore[arg-type]
            return last_response, final_result
        if last_error is not None:
            raise last_error
        # Every viable candidate was saturated → backpressure.
        if saturation_error is not None:
            raise saturation_error
        raise HTTPException(
            status_code=503,
            detail={
                "error": "No model matches constraints",
                "decision": {
                    "requested": ctx.requested_model,
                    "candidates": ctx.candidates,
                    "filtered": ctx.filtered,
                },
            },
        )

    async def route_simple(
        self,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        upstream_path: str = "embeddings",
    ):
        """Minimal non-streaming proxy for simple endpoints (embeddings, rerank,
        and future audio/images). Routes by the requested model/alias/logical id,
        forwards to ``upstream_path``, and retries on retry_on statuses. No prompt
        sizing or max_tokens clamp (those are chat-specific). Returns
        ``(httpx.Response, RouteResult)``; raises HTTPException(503) when no
        candidate, SaturationError when the whole chain is full."""
        headers = headers or {}
        privacy_str = headers.get("X-Llm-Relay-Privacy", "local_only")
        privacy = Privacy(privacy_str if privacy_str in ("local_only", "cloud_ok") else "local_only")
        ctx = RoutingContext(requested_model=request_data.get("model", "") or "", privacy=privacy)
        candidates = self.selector.select_chain(ctx)
        if not candidates:
            raise HTTPException(
                status_code=503,
                detail={"error": "No model matches constraints",
                        "decision": {"requested": ctx.requested_model}},
            )
        retry_codes = {c for c in self.config.policy.fallback.retry_on if c != "connection_error"}
        retry_exceptions = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
        last_response: httpx.Response | None = None
        last_response_candidate: ChainCandidate | None = None
        last_error: Exception | None = None
        saturation_error: SaturationError | None = None
        for candidate in candidates:
            route_result = _candidate_to_route_result(candidate, ctx)
            if not self.discovery.has_free_slot(candidate.backend_key):
                if saturation_error is None:
                    saturation_error = SaturationError(
                        backend_key=candidate.backend_key,
                        retry_after_seconds=candidate.slot_wait_timeout,
                    )
                continue
            try:
                resp = await self.forward_request(
                    candidate.backend_url, candidate.model, request_data,
                    headers=headers, backend_key=candidate.backend_key,
                    slot_wait_timeout=candidate.slot_wait_timeout,
                    upstream_path=upstream_path,
                )
                if str(resp.status_code) in retry_codes:
                    last_response = resp
                    last_response_candidate = candidate
                    continue
                return resp, route_result
            except SaturationError as exc:
                saturation_error = saturation_error or exc
                continue
            except retry_exceptions as exc:
                last_error = exc
                continue
        if last_response is not None:
            return last_response, _candidate_to_route_result(last_response_candidate, ctx)  # type: ignore[arg-type]
        if last_error is not None:
            raise last_error
        if saturation_error is not None:
            raise saturation_error
        raise HTTPException(status_code=503, detail={"error": "No model matches constraints"})


def _candidate_to_route_result(candidate: ChainCandidate, ctx: RoutingContext) -> RouteResult:
    """Build a ``RouteResult`` from a ``ChainCandidate`` for telemetry/response headers."""
    return RouteResult(
        success=True,
        selected_model=candidate.model,
        backend_url=candidate.backend_url,
        provider_name=candidate.provider_name,
        backend_key=candidate.backend_key,
        slot_wait_timeout=candidate.slot_wait_timeout,
        decision={
            "requested": ctx.requested_model,
            "selected": candidate.model,
            "quant": candidate.quant,
            "node": candidate.provider_name,
            "batch": batch_policy_for(ctx.sla_class),
            "sla_class": ctx.sla_class,
            "urgency": ctx.urgency,
            "candidates": ctx.candidates,
            "ranked": ctx.ranked[:5],
            "privacy": ctx.privacy.value,
            # quant is chosen as a side effect of preference (quality) ordering
            # among variants, not an independent cost axis -- see plan 3.
            "trace": "highest-preference variant meeting constraints",
        },
    )


def _estimate_prompt_tokens(request_data: dict) -> int | None:
    """Conservatively estimate the PROMPT's token count for a chat request.

    The relay is provider-agnostic and has no tokenizer, so this approximates
    from character counts and deliberately OVER-estimates: under-counting would
    route a request to a backend too small to hold the prompt (a mid-stream
    failure), whereas over-counting only forgoes spilling to a smaller, faster
    backend. ~3 chars/token over-counts vs. the typical ~3.5-4 for English text.

    Counts message content plus tool/function schemas. ``max_tokens`` is
    deliberately EXCLUDED: it is an output ceiling, not context the model must
    reserve, so it must not gate routing (that conflation pins every request with
    a generous max_tokens to the single largest-context backend). The output is
    fitted separately, per-candidate, by ``_clamp_max_tokens``.

    Returns the estimated prompt tokens (chars / 3), or None when the request is
    trivially small or unparseable — in which case no implicit floor is imposed
    and normal routing applies.
    """
    try:
        messages = request_data.get("messages") or []
        chars = 0
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                # Multimodal content parts: count the text parts' lengths.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chars += len(part["text"])
        # Tool/function schemas are top-level and frequently large (full JSON
        # parameter schemas); tool-using agents are a primary workload, so the
        # definitions must count toward the prompt. Omitting them under-counts —
        # the unsafe direction. Serialize per-spec so a single unserializable
        # entry can't void the message-based estimate.
        for key in ("tools", "functions"):
            spec = request_data.get(key)
            if not spec:
                continue
            try:
                chars += len(json.dumps(spec))
            except (TypeError, ValueError):
                pass
        return (chars // 3) or None
    except Exception:
        return None


def _clamp_max_tokens(request_data: dict, prompt_est: int | None, window: int) -> dict:
    """Cap a request's ``max_tokens`` to the chosen model's remaining headroom.

    ``max_tokens`` is an output ceiling, not a context reservation, so it never
    gates routing (see ``_estimate_prompt_tokens``). But once a model is chosen
    the output still has to fit its window: a request whose ``prompt + max_tokens``
    exceeds the window would overflow it — silently truncated by llama.cpp, hard
    400-rejected by vLLM. So clamp the forwarded ceiling to ``window - prompt``.

    ``prompt_est = chars // 3`` over-counts real tokens, so ``window - prompt_est``
    sits conservatively below the true headroom (the safe direction — no fudge
    factor needed). Returns the request unchanged (same object) when there is
    nothing to clamp; otherwise a shallow copy with the lowered ``max_tokens`` —
    never mutating the caller's dict, which is shared across the candidate chain.
    A clamp can yield a shorter completion than asked (``finish_reason=length``):
    that is honest graceful degradation, far better than excluding the model and
    dead-ending the open-fallthrough in a 503.
    """
    if not prompt_est or not window:
        return request_data
    max_tokens = request_data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return request_data
    headroom = window - prompt_est
    if max_tokens <= headroom:
        return request_data
    return {**request_data, "max_tokens": headroom}
