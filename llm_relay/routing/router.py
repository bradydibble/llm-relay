"""Request routing and upstream forwarding."""
from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import HTTPException

from ..config.loader import ConfigLoader
from ..config.types import Confidentiality, EndpointStatus, NoBackendAvailableError, Privacy, SaturationError
from ..discovery.endpoint import _shared_upstream_bearer, traffic_is_fresh
from ..discovery.manager import DiscoveryManager

_rt_log = logging.getLogger("llm_relay.router")
from .keys import compose_backend_key, compose_backend_url
from .selector import ChainCandidate, ModelSelector, RoutingContext, batch_policy_for


# A model is eligible for a request when it can hold the PROMPT plus this much
# output headroom. The client's max_tokens (an output ceiling) is NOT added to
# the eligibility floor — it is clamped to the chosen model's headroom at forward
# time (see _clamp_max_tokens). So a generous max_tokens neither widens nor pins
# routing; only a prompt that genuinely fits nothing live is refused (oversize).
MIN_OUTPUT_HEADROOM = 1024

# REASONING_OUTPUT_FLOOR is gone (2026-08-11). It silently inflated a client's
# max_tokens (asked 16, got 2048 - measured) to stop chain-of-thought starving
# the answer, back when reasoning shared the content budget invisibly. Reasoning
# is now separated at the source (`set_params: {reasoning_format: deepseek}` on
# llama.cpp models), so a small ceiling gives an honest OpenAI-style outcome:
# finish_reason=length with the reasoning in reasoning_content. A gateway that
# rewrites a client's cost ceiling without saying so betrays the client's
# budget; do not bring the floor back - fix the serve instead.

# DEFAULT_NON_STREAM_MAX_TOKENS: fallback when policy.yaml doesn't set
# default_max_tokens. See PolicyConfig.default_max_tokens for full doc.
DEFAULT_NON_STREAM_MAX_TOKENS = 8192

# Default repetition_penalty for non-streaming requests when the client set
# none. vLLM defaults to 1.0 (no penalty); Ollama defaults to 1.1. Without a
# penalty, certain prompts trigger repetition loops (the 2026-08-16 narf-agent
# hang: a diff-marker C declaration caused the model to repeat "int64 vs
# int64" forever). 1.1 matches Ollama's default and breaks the loop.
DEFAULT_REPETITION_PENALTY = 1.1

# Prompt-size estimation is a heuristic (no server tokenizer). It must be a true
# UPPER bound: under-counting routes a request to a backend too small to hold it
# and the upstream hard-rejects at the boundary (the 2026-07-07 subagent incident:
# a real 66,284-token prompt estimated just under the 65,536 window and 400'd on
# the 9B). Over-counting only forgoes spilling to a smaller/faster backend — the
# safe direction. So the estimator pads the raw char-count heuristic:
#   * PROMPT_CHARS_PER_TOKEN — base ratio; ~3 chars/token is near English but
#     code / structured JSON / CJK / tool payloads pack MORE tokens per char, so 3
#     alone is not a reliable ceiling near the boundary.
#   * PROMPT_ESTIMATE_SAFETY_MARGIN — multiplicative pad so the estimate clears the
#     real count even when the per-char ratio bites. Scales with prompt size, so
#     the absolute headroom grows exactly where boundary risk is highest. 1.2
#     covers a per-char ratio down to ~2.5 (dense code/JSON); the observed incident
#     miss was ~5%, so this is ~4x that. A trivially small prompt still rounds to
#     0 -> None (no implicit floor), preserved by keeping the margin multiplicative.
PROMPT_CHARS_PER_TOKEN = 3
PROMPT_ESTIMATE_SAFETY_MARGIN = 1.2

# Accurate token counting via tiktoken (cl100k_base) when available — a real BPE
# count is within a few percent of the model's own tokenizer AND of what pi/openai
# clients compute, whereas the char heuristic above over-counts English by ~60%.
# That over-count is what pushed pi PAST its own proactive-compaction guard into a
# server 400 it hangs on (cloud never trips this because pi's count matches the
# real window). An accurate count keeps the relay's fit-gate aligned with the
# client's, so a correctly-trimmed request is accepted instead of falsely rejected.
# tiktoken != the qwen/llama tokenizer exactly, but it is far closer than chars/3.
# The char heuristic remains the fallback if tiktoken is unavailable or errors.
_TIKTOKEN_ENC = None
_TIKTOKEN_TRIED = False


def _token_count(text: str) -> int | None:
    """Accurate prompt token count via tiktoken; None if unavailable/errors."""
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if not text:
        return 0
    try:
        if _TIKTOKEN_ENC is None and not _TIKTOKEN_TRIED:
            _TIKTOKEN_TRIED = True
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        if _TIKTOKEN_ENC is None:
            return None
        return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
    except Exception:
        _TIKTOKEN_ENC = None
        return None


def continuous_usage_enabled() -> bool:
    """Whether to ask upstreams for usage on every streamed chunk.

    vLLM supports ``stream_options.continuous_usage_stats``; with it, a stream
    the client aborts still carries exact token counts. Off by default so it is
    probed against the deployed backends before being relied on — see
    LLM_RELAY_CONTINUOUS_USAGE in docs.
    """
    import os

    return os.environ.get("LLM_RELAY_CONTINUOUS_USAGE", "").strip().lower() in ("1", "true", "yes")


def apply_stream_usage_options(body: dict) -> None:
    """Opt into upstream token usage for a streamed request, in place.

    Standard OpenAI streaming omits usage entirely. Caller-supplied values win,
    so a client that deliberately opted out stays opted out.
    """
    existing = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
    opts: dict = {"include_usage": True}
    if continuous_usage_enabled():
        opts["continuous_usage_stats"] = True
    body["stream_options"] = {**opts, **existing}


# Substrings (lowercased) that mark an upstream "prompt exceeds context window"
# rejection across the fleet's backends (llama.cpp `exceed_context_size_error`,
# vLLM "maximum context length", generic proxies). Matched against the response
# body to distinguish a context-overflow 400/413 — which should ESCALATE to a
# larger-window candidate — from an ordinary malformed-request 400, which must not.
_CONTEXT_OVERFLOW_MARKERS = (
    "exceed_context_size",
    "exceeds the available context",
    "maximum context length",
    "context window",
    "context length",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
)


class ContextLengthExceededError(Exception):
    """Raised when a request's prompt exceeds the context window of every model
    that can serve it right now (see ``Selector.diagnose_context_shortfall``).

    The API layer renders ``.body`` as a TOP-LEVEL HTTP 400 ``{"error": {...}}``
    with ``code == "context_length_exceeded"`` — the exact shape OpenAI/Anthropic
    return — so OpenAI-compatible clients (pi, the OpenAI SDK, …) AUTO-COMPACT and
    retry a smaller prompt, instead of reading a 5xx as transient and retrying the
    same oversized prompt until retries are exhausted and the session dies. A bare
    ``HTTPException`` would nest the payload under ``"detail"``, which clients do
    not recognize as a context error."""

    def __init__(self, body: dict):
        super().__init__(body.get("error", {}).get("message", "context_length_exceeded"))
        self.body = body


def _context_length_exceeded_error(shortfall: dict) -> ContextLengthExceededError:
    """Build the OpenAI-standard context_length_exceeded error from a context
    shortfall diagnosis. Reports the currently-servable ceiling so the client
    compacts to a size that will actually route."""
    limit = shortfall.get("max_available_now") or shortfall.get("max_in_catalog") or 0
    est = shortfall.get("estimated_tokens") or 0
    # The message TEXT (not error.code) is what OpenAI-compatible agents pattern-match
    # to trigger auto-compaction: pi keys on /context[_ ]length[_ ]exceeded/i and on the
    # OpenAI phrase "exceeds the model's maximum context length of N". Lead with the
    # literal code AND include the OpenAI phrasing so pi / goose / opencode all recognize
    # it and compact-and-retry instead of hanging on an unrecognized error. See
    # https://github.com/earendil-works/pi-mono/blob/main/packages/ai/src/utils/overflow.ts
    message = (
        f"context_length_exceeded: this request is approximately {est} tokens, which "
        f"exceeds the model's maximum context length of {limit} tokens. Reduce the "
        f"length of the messages (input) and retry."
    )
    return ContextLengthExceededError(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
                # Non-standard diagnostics; OpenAI clients ignore unknown keys.
                "llm_relay_context": shortfall,
            }
        }
    )

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

    def _note_traffic_success(self, backend_key: str | None) -> None:
        """Stamp liveness evidence after a COMPLETED successful real request.
        Promotes the backend and closes breakers — one real 200 outranks every
        probe (observation-first-health-spec §3.1)."""
        client = self.discovery.clients.get(backend_key or "")
        if client is not None:
            client.note_traffic_success()

    def _note_traffic_failure(self, backend_key: str | None) -> None:
        """Stamp a TRANSPORT-level failure of a real request (connect refused/
        timeout, mid-body reset, or backend-minted 502/503/504). Never called
        for 4xx or client aborts — see EndpointClient.note_traffic_failure."""
        client = self.discovery.clients.get(backend_key or "")
        if client is not None:
            client.note_traffic_failure()

    async def _named_live_check(self, named: str, client) -> tuple[bool, dict]:
        """One live availability check for an explicitly named model whose
        cached state says unavailable (spec §3.3): a 503 must reflect an
        attempt made NOW, not a poll that starved half a minute ago.

        Returns ``(recovered, availability)`` where ``availability`` is the
        reason-coded record for the error payload (spec §3.4). On success the
        endpoint state heals in place (status, catalog, breaker) so the caller
        can re-select and dispatch through the one normal path — the live
        check is followed within milliseconds by the real request, which then
        stamps real traffic evidence. Caller holds client.optimistic_lock.
        """
        client.last_live_attempt_ns = time.time_ns()
        avail: dict[str, Any] = {"checked_live": True, "reason": None}
        headers = {"Accept": "application/json"}
        bearer = _shared_upstream_bearer()
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)
            ) as hc:
                resp = await hc.get(
                    f"{client.base_url}{client.health_endpoint}", headers=headers,
                )
            resp.raise_for_status()
            data = resp.json()
            ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        except httpx.ConnectError:
            # Nothing listening: dead — or booting. A backend that served real
            # traffic within the last hour is far more likely mid-restart
            # (gb200 takes ~20 min end to end) than decommissioned; the
            # distinction is advisory, never load-bearing.
            recent = client.last_traffic_success_ns and (
                time.time_ns() - client.last_traffic_success_ns) / 1e9 < 3600
            avail["reason"] = "starting" if recent else "refused"
            if not traffic_is_fresh(client):
                client._record_failure()
            return False, avail
        except httpx.TimeoutException:
            avail["reason"] = "timeout"
            if not traffic_is_fresh(client):
                client._record_failure()
            return False, avail
        except Exception:
            avail["reason"] = "error"
            return False, avail

        if named not in ids:
            # The backend answered and the model genuinely is not loaded
            # (an llm-mode swap, a different serve) — the truthful refusal.
            avail["reason"] = "not_loaded"
            return False, avail
        if client.state.status == EndpointStatus.degraded:
            # Listening and listing ≠ generating: L2 owns the wedge verdict
            # and a live catalog hit must not overrule it (spec invariant 5).
            avail["reason"] = "degraded"
            return False, avail
        # Alive and serving the model: heal L0 state in place, exactly as a
        # successful poll would (the next poll refreshes max_lens etc.).
        client.state.consecutive_failures = 0
        if client.state.circuit_open:
            client.reset_circuit()
        client.consecutive_traffic_failures = 0
        client.state.status = EndpointStatus.healthy
        client.state.models = ids
        _rt_log.info(
            "named-model live check recovered %s on %s (%s) — cached state was stale",
            named, client.provider_name, client.base_url,
        )
        return True, avail

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
                try:
                    resp = await client.post(
                        f"{backend_url}/{upstream_path}",
                        json=body,
                        headers=merged_headers,
                    )
                except httpx.TransportError:
                    # The request never got a well-formed answer — failure
                    # evidence (spec §3.1). 4xx and overflow rejections come
                    # back as responses, not exceptions, and are never stamped.
                    self._note_traffic_failure(backend_key)
                    raise
                # A fully-received 2xx body means the backend generated —
                # liveness evidence. A backend-minted 502/503/504 is transport
                # failure evidence in a status-code costume.
                if 200 <= resp.status_code < 300:
                    self._note_traffic_success(backend_key)
                elif resp.status_code in (502, 503, 504):
                    self._note_traffic_failure(backend_key)
                return resp

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
        apply_stream_usage_options(body)
        merged_headers = {"Content-Type": "application/json", **(headers or {})}
        bearer = _shared_upstream_bearer()
        if bearer and "Authorization" not in merged_headers:
            merged_headers["Authorization"] = f"Bearer {bearer}"
        # 600s per-chunk read timeout — SSE may legitimately stall between tokens
        # on slow models within that window, but a truly dead upstream gets canceled.
        timeout = httpx.Timeout(connect=10.0, read=3600.0, write=10.0, pool=10.0)

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
        except BaseException as exc:
            if isinstance(exc, httpx.TransportError):
                self._note_traffic_failure(backend_key)
            handle.release()
            await client.aclose()
            raise
        if resp.status_code in (502, 503, 504):
            # Backend-minted transport failure, pre-first-byte (spec §3.1).
            self._note_traffic_failure(backend_key)

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
                # Stream drained to the end without error: the backend
                # generated a full response — liveness evidence for the L2
                # probe. An aborted/wedged stream exits via the finally
                # without reaching this.
                if 200 <= resp.status_code < 300:
                    self._note_traffic_success(backend_key)
            except httpx.TransportError:
                # Upstream died mid-stream — failure evidence. A CLIENT abort
                # arrives as GeneratorExit/CancelledError instead and is
                # deliberately not stamped: we chose to stop reading, which
                # says nothing about the backend (spec §3.1).
                self._note_traffic_failure(backend_key)
                raise
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
        # Candidate-lane filter: interactive clients ask for low-latency models,
        # batch jobs ask for high-throughput ones. Models declare their lane via
        # `candidate_lane` in config; requests pass one via this header. Empty /
        # unset means "any lane".
        requested_lane = headers.get("X-Llm-Relay-Candidate-Lane", "").strip() or None
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
            confidentiality=_parse_confidentiality(headers),
            # Derived from the BODY as well as the header: a request that carries
            # tools has declared it needs them, whether or not the client knew our
            # header. Header-only was a real routing bug — a tool-bearing request
            # could open-fallthrough onto a model with no tool support, which then
            # "succeeded" with finish_reason=stop and empty content (measured on
            # qwen3-14b 2026-08-11), the worst failure a client can receive.
            require_tools=(headers.get("X-Llm-Relay-Require-Tools", "false").lower() == "true"
                           or bool(request_data.get("tools"))),
            min_context=max(explicit_min, estimated_min) or None,
            lane=requested_lane,
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
                # Context is the binding constraint: signal it as an OpenAI-standard
                # 400 context_length_exceeded (rendered top-level by the API layer)
                # so clients auto-compact instead of retrying a 5xx to death.
                raise _context_length_exceeded_error(shortfall)
            # Confidentiality is a terminal, actionable block, not an outage: the
            # only models that could serve this request live on hardware CIQ does
            # not own, and the caller did not declare the workload safe to put
            # there. Retrying can never help, so say what to change instead of
            # emitting a generic "no model matches constraints".
            conf_block = self.selector.diagnose_confidentiality_block(ctx)
            if conf_block is not None:
                detail["error"] = "No model matches confidentiality constraints"
                detail["confidentiality"] = conf_block
                raise HTTPException(
                    status_code=503,
                    detail=detail,
                    headers={"X-Llm-Relay-Error": "confidentiality_required"},
                )
            # An EXPLICITLY NAMED model that is not live is terminal and says so,
            # naming the model. Retry-After here would be a guess dressed as a
            # promise — the relay has no idea whether a pinned backend is 5s from
            # returning or has been down for a week — and a generic no-candidate
            # reads as "the fleet is busy", sending callers to fix the wrong thing.
            # Aliases deliberately keep the transient/Retry-After path below: they
            # are open over the fleet, so their members coming back genuinely is
            # the expected remedy, and batch callers rely on that backpressure.
            named = self.selector.explicit_target(ctx)
            if named is not None:
                # Never refuse a named model from cached state (spec §3.3):
                # one live check, guarded so a dead backend costs one connect
                # per 503 storm, not one per request. On recovery, re-select
                # and fall through to the ONE normal dispatch path below —
                # the real request follows the check within milliseconds.
                availability: dict[str, Any] = {"checked_live": False, "reason": None}
                key = self.discovery.model_to_client.get(named)
                client = self.discovery.clients.get(key) if key else None
                if client is None:
                    availability["reason"] = "never_seen"
                elif client.optimistic_lock.locked():
                    availability["reason"] = "check_in_flight"
                    if client.last_live_attempt_ns:
                        availability["last_live_attempt_age_s"] = round(
                            (time.time_ns() - client.last_live_attempt_ns) / 1e9, 1)
                else:
                    async with client.optimistic_lock:
                        recovered, availability = await self._named_live_check(named, client)
                    if recovered:
                        # Reset the decision record so it tells the story of
                        # the post-recovery selection, not the stale one.
                        ctx.candidates = []
                        ctx.filtered = []
                        ctx.ranked = []
                        candidates = self.selector.select_chain(ctx)
                if not candidates:
                    if client is not None and client.last_traffic_success_ns:
                        availability["last_traffic_success_age_s"] = round(
                            (time.time_ns() - client.last_traffic_success_ns) / 1e9, 1)
                    detail["error"] = f"Requested model '{named}' is not available"
                    detail["named_model"] = {
                        "model": named,
                        "status": self.selector.discovery.get_model_state(named).value,
                        "provider": self.config.models.models[named].provider,
                        "availability": availability,
                        "remedy": (
                            f"'{named}' was requested by exact name and is not currently "
                            "serving. The relay does not substitute a different model for "
                            "an explicitly named one. Bring the backend up, or request a "
                            "category alias (e.g. 'main') to route over whatever is live."
                        ),
                    }
                    raise HTTPException(
                        status_code=503,
                        detail=detail,
                        headers={"X-Llm-Relay-Error": "named_model_unavailable"},
                    )
            # Not a context shortfall: if the constraints WOULD be met by a
            # configured model that's merely down/paused right now (a discovery
            # blip or a maintenance pause), the empty chain is a TRANSIENT
            # availability gap — answer with Retry-After backpressure so batch
            # callers wait and retry, instead of a terminal "No model matches
            # constraints". A genuine mismatch (nothing can ever match) stays terminal.
            # Guarded on `named is None`: a named request either raised its own
            # terminal 503 above or RECOVERED via the live check (candidates is
            # now non-empty and dispatch below must run).
            if named is None:
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
            # See the non-streaming loop: once a model overflows the prompt, only
            # a strictly larger-window candidate can serve it.
            overflow_floor = 0
            for candidate in candidates:
                if overflow_floor and candidate.context_window and candidate.context_window <= overflow_floor:
                    continue
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
                    # Apply BOTH defaults to streaming path. max_tokens is
                    # needed here too: without it, a streaming repetition loop
                    # (even with rep_pen=1.1, which reduces but doesn't guarantee
                    # termination) can run for 245K tokens — hours of streamed
                    # garbage consuming a backend slot. A streaming output
                    # budget guarantees termination.
                    _cfg_default = self.config.policy.default_max_tokens
                    _default_mt = _cfg_default if _cfg_default is not None else DEFAULT_NON_STREAM_MAX_TOKENS
                    fwd = _clamp_max_tokens(request_data, prompt_est, candidate.context_window,
                                            default=_default_mt)
                    _cfg_rp = self.config.policy.default_repetition_penalty
                    _default_rp = _cfg_rp if _cfg_rp is not None else DEFAULT_REPETITION_PENALTY
                    fwd = _apply_repetition_penalty_default(fwd, default=_default_rp)
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
                # Context overflow: no SSE bytes have flowed on an error status, so
                # the (small) body can be read to confirm it, and this candidate
                # abandoned for a larger-window one — same escalation as the
                # non-streaming path. Reading DRAINS the streamed body, so a
                # non-overflow 400 that we still commit to must REPLAY the buffered
                # bytes (body_iter would now yield nothing).
                if upstream.status_code in (400, 413):
                    buffered = b""
                    with contextlib.suppress(Exception):
                        buffered = await upstream.aread()
                    if _looks_like_context_overflow(
                        upstream.status_code, buffered.decode("utf-8", "replace")
                    ):
                        overflow_floor = max(overflow_floor, candidate.context_window or 0)
                        await cleanup()
                        continue

                    async def _replay(_data: bytes = buffered, _cleanup=cleanup) -> AsyncIterator[bytes]:
                        try:
                            yield _data
                        finally:
                            await _cleanup()

                    if pending is not None:
                        await pending[3]()
                    return upstream, _replay(), route_result, cleanup
                # Success — or a non-retryable status we must not burn the chain on.
                # Commit to it; free any pending retry-status stream.
                if pending is not None:
                    await pending[3]()
                return upstream, body_iter, route_result, cleanup
            # Chain exhausted.
            if pending is not None:
                # Every candidate gave a retryable status — return the last so the
                # client still sees the upstream 5xx (as the single-candidate path
                # did). Earlier candidates were already cleaned up above.
                return pending
            # Every large-enough candidate overflowed → structured context_overflow.
            if overflow_floor:
                raise _context_overflow_error(ctx, overflow_floor)
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
        # Largest context window that hard-rejected the prompt as too big. Once a
        # model overflows, no candidate with an equal-or-smaller window can serve
        # it, so we escalate PAST them to any larger-window candidate; if none is
        # left the chain surfaces a structured context_overflow (below).
        overflow_floor = 0

        for candidate in candidates:
            # A model no larger than one that already overflowed will overflow too:
            # skip it so escalation only ever moves UP in window size.
            if overflow_floor and candidate.context_window and candidate.context_window <= overflow_floor:
                continue
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
                _cfg_default = self.config.policy.default_max_tokens
                _default_mt = _cfg_default if _cfg_default is not None else DEFAULT_NON_STREAM_MAX_TOKENS
                fwd = _clamp_max_tokens(request_data, prompt_est, candidate.context_window,
                                        default=_default_mt)
                _cfg_rp = self.config.policy.default_repetition_penalty
                _default_rp = _cfg_rp if _cfg_rp is not None else DEFAULT_REPETITION_PENALTY
                fwd = _apply_repetition_penalty_default(fwd, default=_default_rp)
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
                # Prompt overflowed this model's window (estimate under-shot the
                # boundary): escalate to a larger-window candidate rather than
                # returning the raw upstream 400.
                if _looks_like_context_overflow(resp.status_code, resp.text):
                    overflow_floor = max(overflow_floor, candidate.context_window or 0)
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
        # Every candidate large enough was tried and the prompt overflowed them
        # all → structured context_overflow the client can compact-and-retry.
        if overflow_floor:
            raise _context_overflow_error(ctx, overflow_floor)
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
        ctx = RoutingContext(
            requested_model=request_data.get("model", "") or "",
            privacy=privacy,
            confidentiality=_parse_confidentiality(headers),
        )
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


def _parse_confidentiality(headers: dict) -> Confidentiality:
    """Parse the caller's workload-sensitivity declaration. Fails CLOSED.

    Only the exact token ``non_confidential`` opts a request into third-party
    hardware. Absent, empty, misspelled, or unrecognized values all resolve to
    ``confidential``. A data-governance control must never widen the hardware
    pool because of a typo — the quiet failure has to be the safe one.
    """
    raw = (headers.get("X-Llm-Relay-Confidentiality") or "").strip().lower()
    return (
        Confidentiality.non_confidential
        if raw == Confidentiality.non_confidential.value
        else Confidentiality.confidential
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
            "quant": candidate.quant,
            "node": candidate.provider_name,
            "batch": batch_policy_for(ctx.sla_class),
            "sla_class": ctx.sla_class,
            "urgency": ctx.urgency,
            "candidates": ctx.candidates,
            "ranked": ctx.ranked[:5],
            "privacy": ctx.privacy.value,
            "confidentiality": ctx.confidentiality.value,
            # quant is chosen as a side effect of preference (quality) ordering
            # among variants, not an independent cost axis -- see plan 3.
            "trace": "highest-preference variant meeting constraints",
        },
    )


def _looks_like_context_overflow(status_code: int, body_text: str | None) -> bool:
    """True when an upstream 400/413 is a 'prompt exceeds context window' reject
    (as opposed to an ordinary malformed-request 400).

    Drives the escalate-to-a-larger-window backstop in ``route_and_forward``: when
    the prompt estimate under-shot and the chosen model hard-rejects at its
    boundary, the relay retries a bigger-window candidate instead of surfacing the
    raw upstream 400. Matched on the response body against
    ``_CONTEXT_OVERFLOW_MARKERS`` so it never mistakes a genuine bad request for an
    overflow (which would burn the chain pointlessly)."""
    if status_code not in (400, 413):
        return False
    b = (body_text or "").lower()
    return any(m in b for m in _CONTEXT_OVERFLOW_MARKERS)


def _context_overflow_error(ctx: RoutingContext, largest_window_tried: int) -> HTTPException:
    """Structured, client-reconcilable error for a prompt that overflowed every
    routable model. Carries an ``X-Llm-Relay-Error`` header and a ``context`` block
    (mirroring ``diagnose_context_shortfall``) so a harness can compact/resize and
    retry deterministically instead of parsing a backend's freeform 400 text."""
    return HTTPException(
        status_code=413,
        detail={
            "error": "context_overflow",
            "message": (
                "Prompt exceeds the context window of every model this request "
                "could route to. Reduce the prompt (compact/summarize) and retry."
            ),
            "context": {
                "estimated_tokens": ctx.min_context,
                "largest_window_tried": largest_window_tried,
                "classification": "oversize_period",
            },
            "decision": {
                "requested": ctx.requested_model,
                "candidates": ctx.candidates,
            },
        },
        headers={"X-Llm-Relay-Error": "context_overflow"},
    )


def _estimate_prompt_tokens(request_data: dict) -> int | None:
    """Conservatively estimate the PROMPT's token count for a chat request.

    The relay is provider-agnostic and has no tokenizer, so this approximates
    from character counts and deliberately OVER-estimates: under-counting would
    route a request to a backend too small to hold the prompt (a boundary
    hard-reject), whereas over-counting only forgoes spilling to a smaller, faster
    backend. The estimate is padded to a true upper bound — base char ratio times a
    safety margin (see the PROMPT_* constants above) — because a bare ~3 chars/token
    ratio under-counts for code / JSON / CJK / tool payloads and missed a boundary
    by ~5% in the 2026-07-07 subagent incident.

    Counts message content plus tool/function schemas. ``max_tokens`` is
    deliberately EXCLUDED: it is an output ceiling, not context the model must
    reserve, so it must not gate routing (that conflation pins every request with
    a generous max_tokens to the single largest-context backend). The output is
    fitted separately, per-candidate, by ``_clamp_max_tokens``.

    Returns the upper-bound estimated prompt tokens, or None when the request is
    trivially small or unparseable — in which case no implicit floor is imposed
    and normal routing applies.
    """
    try:
        parts: list[str] = []
        messages = request_data.get("messages") or []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Multimodal content parts: count the text parts.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
        # Tool/function schemas are top-level and frequently large (full JSON
        # parameter schemas); tool-using agents are a primary workload, so the
        # definitions must count toward the prompt. Serialize per-spec so a single
        # unserializable entry can't void the estimate.
        for key in ("tools", "functions"):
            spec = request_data.get(key)
            if not spec:
                continue
            try:
                parts.append(json.dumps(spec))
            except (TypeError, ValueError):
                pass
        chars = sum(len(p) for p in parts)
        if not chars:
            return None
        # Prefer an accurate BPE count (tiktoken) with NO safety pad — it tracks the
        # real token count within a few percent, so it neither over-rejects a
        # client's correctly-trimmed prompt (the pi-hang trigger) nor under-counts
        # into a boundary hard-reject (the 2026-07-07 incident, which the accurate
        # count also catches). Fall back to the padded char heuristic only when
        # tiktoken is unavailable/errors. A trivially small prompt -> None.
        exact = _token_count("\n".join(parts))
        if exact is not None:
            return exact or None
        return int(chars / PROMPT_CHARS_PER_TOKEN * PROMPT_ESTIMATE_SAFETY_MARGIN) or None
    except Exception:
        return None


def _clamp_max_tokens(request_data: dict, prompt_est: int | None, window: int,
                      default: int | None = None) -> dict:
    """Fit a request's ``max_tokens`` to the chosen model: cap DOWN to the window
    headroom. Never raises the ceiling - a client's max_tokens is a cost and
    latency budget, and inflating it silently is a lie about money (the old
    reasoning floor did exactly that; see the tombstone at the top of this file).

    ``max_tokens`` is an output ceiling, not a context reservation, so it never
    gates routing (see ``_estimate_prompt_tokens``). But once a model is chosen
    the output still has to fit its window: a request whose ``prompt + max_tokens``
    exceeds the window would overflow it - silently truncated by llama.cpp, hard
    400-rejected by vLLM. So the ceiling is capped to ``window - prompt``.

    ``default``: when the client set NO ``max_tokens`` (None or <=0) and
    ``default`` is provided, it is applied as the ceiling. This is NOT
    inflation (the client set nothing); it prevents unbounded non-streaming
    generation (vLLM defaults to max_model_len - prompt = hours on large-context
    models). Only the non-streaming call site passes ``default``; streaming
    leaves it unset (the client sees tokens and can disconnect).

    Returns the request unchanged (same object) when the cap doesn't move
    ``max_tokens``; otherwise a shallow copy - never mutating the caller's dict,
    which is shared across the candidate chain. A down-clamp can still yield a
    shorter completion than asked (``finish_reason=length``): honest graceful
    degradation, far better than dead-ending the fallthrough in a 503.
    """
    max_tokens = request_data.get("max_tokens")
    has_client_ceiling = isinstance(max_tokens, int) and max_tokens > 0
    if not has_client_ceiling:
        # default=0 or default=None means "no default cap" — return unchanged.
        # Only a positive default applies a ceiling.
        if not default or default <= 0:
            return request_data
        max_tokens = default
    headroom = (window - prompt_est) if (prompt_est and window) else None
    if headroom is not None and max_tokens > headroom:
        return {**request_data, "max_tokens": headroom}
    if has_client_ceiling:
        return request_data
    return {**request_data, "max_tokens": max_tokens}


def _apply_repetition_penalty_default(
    request_data: dict, default: float | None = None
) -> dict:
    """Apply a ``repetition_penalty`` floor when the client set none or a
    lower value.

    vLLM's default is 1.0 (no penalty); Ollama's is 1.1. Without a penalty,
    certain prompts trigger repetition loops where the model generates the
    same token sequence indefinitely (the 2026-08-16 narf-agent hang: a
    diff-marker C declaration caused the model to repeat "int64 vs int64"
    forever). 1.1 matches Ollama's default and breaks the loop.

    This is a SAFETY FLOOR, not just a default: a client that explicitly
    sends repetition_penalty=1.0 (or null) gets clamped up to the floor.
    Clients that send a higher value (e.g. 1.2) are forwarded unchanged.
    This prevents clients that serialize 1.0 as their default from
    bypassing the protection (ChatGPT gpt-5.6 advised: 'many clients
    serialize defaults such as 1.0; a presence-only default may protect
    fewer requests than expected').

    Returns the same object when no change is needed; otherwise a shallow copy.
    """
    if not default or default <= 0:
        return request_data
    existing = request_data.get("repetition_penalty")
    if existing is None or not isinstance(existing, (int, float)) or existing < default:
        return {**request_data, "repetition_penalty": default}
    return request_data
