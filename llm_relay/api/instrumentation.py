"""Optional OTel capture for llm-relay → Phoenix.

Disabled by default. Enable with `LLM_RELAY_TELEMETRY=1`. The exporter
runs in a background thread (BatchSpanProcessor) and is fire-and-forget;
the request path never blocks on it, and an exporter failure is silent.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any

from .. import metrics

_TRACER: Any = None
_INITIALIZED = False

# Conservative secret/PII patterns redacted from captured prompts and completions.
# Goal: avoid storing live credentials in Phoenix. False positives are preferred over leaks.
_REDACT_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),          "<anthropic_key>"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"),                  "<openai_key>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"),                     "<aws_access_key>"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"),                 "<github_token>"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"),                 "<github_oauth>"),
    (re.compile(r"glpat-[A-Za-z0-9_\-]{20,}"),            "<gitlab_token>"),
    (re.compile(r"xox[bpars]-[A-Za-z0-9\-]{10,}"),         "<slack_token>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "<private_key>"),
    # Generic Bearer / Authorization header values
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._\-]{20,}"), r"\1<token>"),
    # Anthropic API key header form sometimes appears verbatim in pasted curl
    (re.compile(r"(?i)(x-api-key\s*:\s*)[A-Za-z0-9_\-]{20,}"), r"\1<key>"),
]


def _redact(s: str) -> str:
    if not s:
        return s
    out = s
    for pat, repl in _REDACT_PATTERNS:
        out = pat.sub(repl, out)
    return out


def is_enabled() -> bool:
    return os.environ.get("LLM_RELAY_TELEMETRY", "0").lower() in {"1", "true", "yes", "on"}


def _init_tracer() -> Any:
    global _TRACER, _INITIALIZED
    if _INITIALIZED:
        return _TRACER
    _INITIALIZED = True
    if not is_enabled():
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.environ.get("LLM_RELAY_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
        project = os.environ.get("PHOENIX_PROJECT_NAME", "llm-relay")
        resource = Resource.create({"service.name": "llm-relay", "openinference.project.name": project})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("llm_relay.api")
    except Exception as e:
        print(f"[llm-relay] telemetry init failed, disabling: {e}", file=sys.stderr)
        _TRACER = None
    return _TRACER


def _classify_stream_outcome(status: int, exc: BaseException | None, finished: bool) -> str:
    """Outcome label for a streamed response, based on how it ACTUALLY terminated.

    Priority (first match wins):
      - error HTTP status (>=400) -> ``upstream_error``: the response itself was an
        error regardless of how its body drained (e.g. the all-retryable-5xx stream
        the spill path returns, which has no ``[DONE]``).
      - client cancellation (``CancelledError``/``GeneratorExit``) ->
        ``client_disconnect``: the client hung up mid-stream — not the backend's
        fault, so it must not count against backend reliability.
      - any other exception -> ``stream_error``: the backend dropped mid-stream.
      - clean finish (``[DONE]`` or a non-null ``finish_reason`` was seen) ->
        ``success``.
      - otherwise -> ``stream_incomplete``: ended early with no terminal marker
        (the silent hangup that was previously mislabeled ``success``).
    """
    if status >= 400:
        return "upstream_error"
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return "client_disconnect"
    if exc is not None:
        return "stream_error"
    return "success" if finished else "stream_incomplete"


def sse_finished(raw: bytes) -> bool:
    """True if the SSE stream terminated cleanly — a ``[DONE]`` sentinel or a
    non-null ``finish_reason`` on any choice was seen. False means the stream was
    cut off before any terminal marker (a truncated / aborted generation)."""
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("data:"):
            continue
        payload = s[5:].strip()
        if payload == "[DONE]":
            return True
        try:
            j = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for ch in j.get("choices") or []:
            if ch.get("finish_reason") is not None:
                return True
    return False


def reassemble_sse(raw: bytes) -> tuple[str, dict]:
    """Reassemble llama.cpp SSE chat-completion stream into (text, usage).

    Captures both `delta.content` (visible assistant text) and
    `delta.reasoning_content` (Qwen-style chain-of-thought). The returned
    text concatenates content only; reasoning is folded into usage under
    a `_reasoning_content` key so the caller can surface it separately.

    Also reports `_frame_count` (how many delta frames carried content or
    reasoning) and `_saw_incremental` (usage arrived on a chunk that also
    carried a delta, i.e. vLLM ``continuous_usage_stats``). Both exist so an
    aborted stream — which never receives the terminal usage chunk — can still
    be counted instead of silently recorded as zero tokens.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    frame_count = 0
    saw_incremental = False
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("data:"):
            continue
        payload = s[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for ch in j.get("choices") or []:
            delta = ch.get("delta") or {}
            c = delta.get("content")
            if c:
                content_parts.append(c)
            rc = delta.get("reasoning_content")
            if rc:
                reasoning_parts.append(rc)
            if c or rc:
                frame_count += 1
        if j.get("usage"):
            if j.get("choices"):
                # Usage alongside a content delta = continuous_usage_stats.
                saw_incremental = True
            usage = j["usage"]
        # llama.cpp emits non-standard `timings` on the final chunk with token counts;
        # use them as a fallback when standard `usage` isn't present (stream w/o include_usage).
        t = j.get("timings") or {}
        if t.get("predicted_n") and "completion_tokens" not in usage:
            usage["completion_tokens"] = int(t["predicted_n"])
        if t.get("prompt_n") and "prompt_tokens" not in usage:
            usage["prompt_tokens"] = int(t["prompt_n"])
    if "completion_tokens" in usage and "prompt_tokens" in usage and "total_tokens" not in usage:
        usage["total_tokens"] = usage["completion_tokens"] + usage["prompt_tokens"]
    if reasoning_parts:
        usage["_reasoning_content"] = "".join(reasoning_parts)
    usage["_frame_count"] = frame_count
    usage["_saw_incremental"] = saw_incremental
    return "".join(content_parts), usage


def request_shape(request_body: dict | None) -> dict:
    """Structural fingerprint of a request — counts and hashes, never content.

    ``prefix_hash`` covers every message except the last, so a conversation
    resent turn after turn is recognisable across requests. That is what makes
    prompt-cache opportunity measurable without storing any text.
    """
    import hashlib

    body = request_body if isinstance(request_body, dict) else {}
    messages = body.get("messages")
    messages = messages if isinstance(messages, list) else []

    def _digest(parts) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(repr(p).encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    system_parts = [m.get("content") for m in messages
                    if isinstance(m, dict) and m.get("role") == "system"]
    prefix_parts = [(m.get("role"), m.get("content")) for m in messages[:-1]
                    if isinstance(m, dict)]
    tools = body.get("tools")

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    return {
        "message_count": len(messages),
        "system_hash": _digest(system_parts) if system_parts else None,
        "prefix_hash": _digest(prefix_parts) if prefix_parts else None,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "temperature": float(temperature) if isinstance(temperature, (int, float)) else None,
        "max_tokens": int(max_tokens) if isinstance(max_tokens, int) else None,
    }


def completion_text(response_text: str | None, response_body: dict | None,
                    usage: dict | None) -> tuple[str, str]:
    """``(assistant_text, reasoning_text)`` for one completion, either shape.

    The two response paths deliver the answer differently: the streaming path
    hands over reassembled ``response_text`` plus ``usage["_reasoning_content"]``
    (``app.py:1549``), while the non-streaming path hands over a parsed
    ``response_body`` and ``response_text=None`` (``app.py:1585-1589``). Reading
    only ``response_text`` would therefore archive every non-streamed request as
    a prompt with no answer -- the majority of requests, silently half-captured.

    Never raises: a malformed body yields empty strings.
    """
    reasoning = str((usage or {}).get("_reasoning_content") or "")
    if response_text:
        return response_text, reasoning
    if not isinstance(response_body, dict):
        return "", reasoning
    parts: list[str] = []
    for choice in response_body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            # Multimodal output parts: keep the structure rather than lose it.
            parts.append(json.dumps(content))
        if not reasoning:
            rc = message.get("reasoning_content")
            if isinstance(rc, str) and rc:
                reasoning = rc
    return "\n".join(parts), reasoning


def emit_chat_completion(
    *,
    request_body: dict,
    response_body: dict | None,
    response_text: str | None,
    usage: dict | None,
    model_resolved: str | None,
    provider_name: str | None,
    user_agent: str,
    start_ns: int,
    end_ns: int,
    status_code: int,
    streamed: bool,
    error: str | None = None,
    outcome: str = "success",
    client: str | None = None,
    fell_back: bool = False,
    ttft_ns: int | None = None,
    principal: str | None = None,
    confidentiality: str | None = None,
) -> None:
    """Emit telemetry for one chat completion: Prometheus metrics (always) and,
    when LLM_RELAY_TELEMETRY is enabled, an OpenInference span. Best-effort;
    never raises into the request path."""
    # A non-streamed llama.cpp response carries server-side prefill time in
    # `timings.prompt_ms`. When the caller didn't measure an end-to-end value (the
    # streaming path passes a real first-chunk ttft_ns; the non-streaming path
    # passes none), fall back to prompt_ms as the TTFT proxy: prefill dominates
    # TTFT on this fleet, so it's a faithful stand-in and it keeps the aggregate
    # ttft histogram from being streaming-only. Flows to both the Prometheus
    # histogram and the span's TTFT attribute below. Backends that don't report
    # timings (e.g. vLLM) leave ttft_ns None — no fabricated value.
    if ttft_ns is None and isinstance(response_body, dict):
        _pm = (response_body.get("timings") or {}).get("prompt_ms")
        if isinstance(_pm, (int, float)) and _pm >= 0:
            ttft_ns = int(_pm * 1e6)
    # Resolve the token counts ONCE, here, so Prometheus and the durable store
    # can never disagree about what a request cost.
    from ..usage_math import resolve_usage

    eff_usage = usage if usage else None
    counts = resolve_usage(
        usage=eff_usage,
        response_body=response_body,
        streamed=streamed,
        frame_count=int((eff_usage or {}).get("_frame_count") or 0),
        content_text=response_text or "",
        reasoning_text=str((eff_usage or {}).get("_reasoning_content") or ""),
        saw_incremental=bool((eff_usage or {}).get("_saw_incremental")),
    )

    # Metrics first, independent of the OTLP tracer (Phoenix may be down).
    try:
        duration_s = max(0.0, (end_ns - start_ns) / 1e9)
        metrics.get_metrics().record_request(
            alias=request_body.get("model") if isinstance(request_body, dict) else None,
            model=model_resolved,
            provider=provider_name,
            outcome=outcome,
            client=client,
            usage=usage,
            response_body=response_body,
            duration_s=duration_s,
            fell_back=fell_back,
            ttft_s=(ttft_ns / 1e9) if ttft_ns is not None else None,
            principal=principal,
            counts=counts,
        )
    except Exception as e:
        print(f"[llm-relay] metrics record failed (ignored): {e}", file=sys.stderr)

    # One id per completion event, generated before either durable write, so a
    # usage row can be joined to the prompt row describing the same request.
    # Deliberately not inside the store blocks below: two uuid4() calls would
    # produce two unrelated ids and silently break that join.
    import uuid as _uuid

    request_id = _uuid.uuid4().hex

    # Durable row. Best-effort: a storage problem must never surface here.
    try:
        from ..usage_store import get_store

        store = get_store()
        if store is not None:
            import datetime as _dt

            ts = end_ns / 1_000_000_000
            shape = request_shape(request_body)
            store.record({
                "request_id": request_id,
                "ts": ts,
                "day": _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d"),
                "principal": principal or "anonymous",
                "client": client or "unknown",
                "alias": (request_body or {}).get("model"),
                "model": model_resolved or "none",
                "provider": provider_name or "",
                "outcome": outcome,
                "streamed": 1 if streamed else 0,
                "duration_ms": int((end_ns - start_ns) / 1_000_000),
                "ttft_ms": int(ttft_ns / 1_000_000) if ttft_ns else None,
                "input_tokens": counts.input_tokens,
                "output_tokens": counts.output_tokens,
                "reasoning_tokens": counts.reasoning_tokens,
                "cache_read_tokens": counts.cache_read_tokens,
                "usage_source": counts.usage_source,
                "reasoning_source": counts.reasoning_source,
                "synthetic": 0,
                "message_count": shape["message_count"],
                "system_hash": shape["system_hash"],
                "prefix_hash": shape["prefix_hash"],
                "tool_count": shape["tool_count"],
                "temperature": shape["temperature"],
                "max_tokens": shape["max_tokens"],
                "confidentiality": confidentiality,
                "fell_back": 1 if fell_back else 0,
            })
    except Exception:
        pass

    # Prompt/completion content. Separate store, separate env flag, and the same
    # best-effort contract: a capture problem must never surface in a request.
    # ``LLM_RELAY_PROMPT_DB`` unset means get_store() returns None and nothing
    # here touches the disk -- which is how this ships.
    try:
        from ..prompt_store import get_store as _get_prompt_store

        pstore = _get_prompt_store()
        if pstore is not None:
            import datetime as _dt

            _ts = end_ns / 1_000_000_000
            _msgs = (request_body or {}).get("messages")
            _msgs = _msgs if isinstance(_msgs, list) else []
            _completion, _reasoning = completion_text(
                response_text, response_body, usage)
            pstore.record({
                "request_id": request_id,
                "ts": _ts,
                "day": _dt.datetime.fromtimestamp(
                    _ts, _dt.timezone.utc).strftime("%Y-%m-%d"),
                "principal": principal or "anonymous",
                "client": client or "unknown",
                "model": model_resolved or "none",
                # ``content`` is passed through untouched: it may be a list of
                # multimodal parts, and the store's _as_text keeps the text
                # parts while reducing an image or audio part to a marker.
                # str()-ing it here would archive base64 payloads instead.
                "messages": [
                    {"role": str(m.get("role") or ""), "content": m.get("content")}
                    for m in _msgs if isinstance(m, dict)
                ],
                "completion": _completion,
                "reasoning": _reasoning,
            })
    except Exception:
        pass

    tracer = _init_tracer()
    if tracer is None:
        return
    try:
        span = tracer.start_span("chat_completion", start_time=start_ns)
        try:
            # Attribute ORDER matters. OpenTelemetry caps a span at
            # SpanLimits.max_attributes (default 128) and, when full, evicts the
            # OLDEST attribute. The per-message breakdown below is unbounded (two
            # attributes per chat message), so on a long conversation it would push
            # whatever was set first off the span. We therefore set the
            # high-cardinality, lower-value per-message attributes FIRST and the
            # critical, low-cardinality routing attributes (model / provider /
            # outcome / span.kind) LAST, so the latter always survive — otherwise a
            # 60-message request silently loses its model and provider and renders as
            # an untyped "unknown" span. The full prompt and response are also kept
            # verbatim in input.value / output.value, which never depend on the
            # per-message keys.
            for i, msg in enumerate(request_body.get("messages") or []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content)
                span.set_attribute(f"llm.input_messages.{i}.message.role", str(role))
                span.set_attribute(f"llm.input_messages.{i}.message.content", _redact(str(content))[:32000])

            out_usage = usage or {}
            output_value = None
            if response_body and isinstance(response_body, dict):
                for i, ch in enumerate(response_body.get("choices") or []):
                    m = ch.get("message") or {}
                    span.set_attribute(f"llm.output_messages.{i}.message.role", str(m.get("role", "assistant")))
                    span.set_attribute(f"llm.output_messages.{i}.message.content", _redact(str(m.get("content", "")))[:32000])
                out_usage = response_body.get("usage") or out_usage
                output_value = _redact(json.dumps(response_body))[:64000]
            elif response_text is not None:
                span.set_attribute("llm.output_messages.0.message.role", "assistant")
                span.set_attribute("llm.output_messages.0.message.content", _redact(response_text)[:32000])
                output_value = _redact(response_text)[:64000]

            # --- critical, low-cardinality attributes: set LAST so eviction can
            # never drop them, regardless of how long the conversation is ---
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.relay.outcome", outcome)
            if model_resolved:
                span.set_attribute("llm.model_name", model_resolved)
            if provider_name:
                span.set_attribute("llm.provider", provider_name)
            if ttft_ns is not None:
                span.set_attribute("llm.latency.time_to_first_token_seconds", ttft_ns / 1e9)

            invocation = {
                k: v for k, v in request_body.items()
                if k in {"temperature", "top_p", "max_tokens", "stream", "stop", "n", "presence_penalty", "frequency_penalty"}
            }
            span.set_attribute("llm.invocation_parameters", json.dumps(invocation))

            if output_value is not None:
                span.set_attribute("output.value", output_value)
            if out_usage.get("_reasoning_content"):
                span.set_attribute("llm.reasoning_content", _redact(str(out_usage["_reasoning_content"]))[:32000])
            if "prompt_tokens" in out_usage:
                span.set_attribute("llm.token_count.prompt", int(out_usage["prompt_tokens"]))
            if "completion_tokens" in out_usage:
                span.set_attribute("llm.token_count.completion", int(out_usage["completion_tokens"]))
            if "total_tokens" in out_usage:
                span.set_attribute("llm.token_count.total", int(out_usage["total_tokens"]))

            span.set_attribute("input.value", _redact(json.dumps({"messages": request_body.get("messages") or []}))[:64000])
            span.set_attribute("user_agent", user_agent or "")
            span.set_attribute("http.status_code", int(status_code))
            span.set_attribute("llm.streamed", bool(streamed))
            if error:
                from opentelemetry.trace import Status, StatusCode
                span.set_attribute("error.message", error[:8000])
                span.set_status(Status(StatusCode.ERROR, error[:512]))
        finally:
            span.end(end_time=end_ns)
    except Exception as e:
        print(f"[llm-relay] emit_chat_completion failed (ignored): {e}", file=sys.stderr)
