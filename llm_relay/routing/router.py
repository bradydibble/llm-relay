"""Request routing and upstream forwarding."""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import HTTPException

from ..config.loader import ConfigLoader
from ..config.types import Privacy, SaturationError
from ..discovery.endpoint import _shared_upstream_bearer
from ..discovery.manager import DiscoveryManager
from .keys import compose_backend_key, compose_backend_url
from .selector import ChainCandidate, ModelSelector, RoutingContext


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

    async def forward_request(
        self,
        backend_url: str,
        model_name: str,
        request_data: dict[str, Any],
        headers: dict[str, str] | None = None,
        backend_key: str | None = None,
        slot_wait_timeout: float = 30.0,
    ) -> httpx.Response:
        body = dict(request_data)
        body["model"] = model_name
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        # Authenticate to api-key'd upstreams using the shared homelab bearer
        # (see endpoint.py for env-var resolution). Caller-provided
        # Authorization wins so a future per-request auth path can override.
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # backend_key="" / None → acquire_slot is a no-op (no semaphore registered).
        async with self.discovery.acquire_slot(backend_key or "", wait_timeout=slot_wait_timeout):
            async with httpx.AsyncClient(timeout=300.0) as client:
                return await client.post(
                    f"{backend_url}/chat/completions",
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
        body = dict(request_data)
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

        weights: dict[str, float] = {}
        weights_str = headers.get("X-Llm-Relay-Weights", "")
        if weights_str:
            for pair in weights_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        weights[k.strip()] = float(v.strip())
                    except ValueError:
                        pass

        # Context-aware routing: an explicit X-Llm-Relay-Min-Context header is a
        # floor the caller asserts; we also estimate the request's own context
        # need from its body and take the larger. The selector drops candidates
        # whose context_window is below min_context, so a large request is never
        # routed to a backend too small to hold it (which would fail mid-stream).
        explicit_min = int(headers.get("X-Llm-Relay-Min-Context", "0") or 0)
        estimated_min = _estimate_request_min_context(request_data) or 0
        ctx = RoutingContext(
            requested_model=request_data.get("model", "") or "",
            privacy=privacy,
            weights=weights,
            require_tools=headers.get("X-Llm-Relay-Require-Tools", "false").lower() == "true",
            min_context=max(explicit_min, estimated_min) or None,
        )

        candidates = self.selector.select_chain(ctx)
        if not candidates:
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

        # Streaming: route once, no retry (slot lifecycle prevents safe retry).
        if stream:
            candidate = candidates[0]
            route_result = _candidate_to_route_result(candidate, ctx)
            upstream, body_iter, cleanup = await self.stream_request(
                candidate.backend_url, candidate.model, request_data,
                headers=headers,
                backend_key=candidate.backend_key,
                slot_wait_timeout=candidate.slot_wait_timeout,
            )
            return upstream, body_iter, route_result, cleanup

        # Non-streaming: walk the chain, retry on retry_on errors.
        last_response: httpx.Response | None = None
        last_response_candidate: ChainCandidate | None = None
        last_error: Exception | None = None

        for candidate in candidates:
            route_result = _candidate_to_route_result(candidate, ctx)
            try:
                resp = await self.forward_request(
                    candidate.backend_url, candidate.model, request_data,
                    headers=headers,
                    backend_key=candidate.backend_key,
                    slot_wait_timeout=candidate.slot_wait_timeout,
                )
                if str(resp.status_code) in retry_codes:
                    last_response = resp
                    last_response_candidate = candidate
                    continue
                return resp, route_result
            except SaturationError:
                # Slot saturation is backpressure, not backend failure.
                # Propagate immediately so the caller emits Retry-After.
                raise
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
            "candidates": ctx.candidates,
            "ranked": ctx.ranked[:5],
            "privacy": ctx.privacy.value,
        },
    )


def _estimate_request_min_context(request_data: dict) -> int | None:
    """Conservatively estimate the context window a chat request needs.

    The relay is provider-agnostic and has no tokenizer, so this approximates
    from character counts and deliberately OVER-estimates: under-estimating
    would route a request to a backend too small to hold it (a mid-stream
    failure), whereas over-estimating only forgoes spilling to a smaller, faster
    backend. ~3 chars/token over-counts vs. the typical ~3.5-4 for English text.

    Returns the estimated token budget (prompt chars / 3, plus any requested
    ``max_tokens``), or None when the request is trivially small or unparseable —
    in which case no implicit floor is imposed and normal routing applies.
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
        # definitions must count toward the floor. Omitting them under-counts —
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
        est = chars // 3
        max_tokens = request_data.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            est += max_tokens
        return est or None
    except Exception:
        return None
