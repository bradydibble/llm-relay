"""Optional OTel capture for llm-relay → Phoenix.

Disabled by default. Enable with `LLM_RELAY_TELEMETRY=1`. The exporter
runs in a background thread (BatchSpanProcessor) and is fire-and-forget;
the request path never blocks on it, and an exporter failure is silent.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

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


def reassemble_sse(raw: bytes) -> tuple[str, dict]:
    """Reassemble llama.cpp SSE chat-completion stream into (text, usage).

    Captures both `delta.content` (visible assistant text) and
    `delta.reasoning_content` (Qwen-style chain-of-thought). The returned
    text concatenates content only; reasoning is folded into usage under
    a `_reasoning_content` key so the caller can surface it separately.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
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
        if j.get("usage"):
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
    return "".join(content_parts), usage


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
) -> None:
    """Emit one OpenInference-style LLM span. Best-effort; never raises."""
    tracer = _init_tracer()
    if tracer is None:
        return
    try:
        span = tracer.start_span("chat_completion", start_time=start_ns)
        try:
            span.set_attribute("openinference.span.kind", "LLM")
            if model_resolved:
                span.set_attribute("llm.model_name", model_resolved)
            if provider_name:
                span.set_attribute("llm.provider", provider_name)

            invocation = {
                k: v for k, v in request_body.items()
                if k in {"temperature", "top_p", "max_tokens", "stream", "stop", "n", "presence_penalty", "frequency_penalty"}
            }
            span.set_attribute("llm.invocation_parameters", json.dumps(invocation))

            for i, msg in enumerate(request_body.get("messages") or []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content)
                span.set_attribute(f"llm.input_messages.{i}.message.role", str(role))
                span.set_attribute(f"llm.input_messages.{i}.message.content", _redact(str(content))[:32000])

            out_usage = usage or {}
            if response_body and isinstance(response_body, dict):
                for i, ch in enumerate(response_body.get("choices") or []):
                    m = ch.get("message") or {}
                    span.set_attribute(f"llm.output_messages.{i}.message.role", str(m.get("role", "assistant")))
                    span.set_attribute(f"llm.output_messages.{i}.message.content", _redact(str(m.get("content", "")))[:32000])
                out_usage = response_body.get("usage") or out_usage
                span.set_attribute("output.value", _redact(json.dumps(response_body))[:64000])
            elif response_text is not None:
                span.set_attribute("llm.output_messages.0.message.role", "assistant")
                span.set_attribute("llm.output_messages.0.message.content", _redact(response_text)[:32000])
                span.set_attribute("output.value", _redact(response_text)[:64000])

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
