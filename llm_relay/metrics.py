"""Prometheus metrics for llm-relay.

Decoupled from the OTLP span path (``instrumentation.py``): these counters and
gauges record request / token / fallback activity plus per-backend health, and
are exposed at ``/metrics`` regardless of whether telemetry (Phoenix) is
enabled or reachable.

Design notes:
- Recording is best-effort and must never raise into the request path. The
  caller wraps ``record_request`` in try/except; label values are coerced to
  safe, bounded strings here (None -> "none", unknown clients -> "other").
- Backend health gauges are *pull-based*: ``DiscoveryCollector`` reads live
  state off the ``DiscoveryManager`` at scrape time, so the discovery poll loop
  is never touched.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, Counter, Histogram, disable_created_metrics, generate_latest
from prometheus_client.core import GaugeMetricFamily

# Calling agents we attribute traffic to. Anything else buckets to "other" so a
# malformed or novel header value can't explode label cardinality.
_KNOWN_CLIENTS = {"claude-code", "agent-a", "agent-b", "agent-c"}

# End-to-end latency buckets (seconds): sub-second routing overhead through
# multi-minute large-model generations on the local fleet.
_DURATION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf"))

# Dedicated registry for relay metrics — kept off the global default REGISTRY so
# repeated create_app() calls (tests, reloads) never collide, and the relay's
# series stay isolated from any future default collectors.
RELAY_REGISTRY = CollectorRegistry()

# Drop the per-counter _created timestamp series — halves counter cardinality
# and the create-time isn't useful for these ops metrics.
disable_created_metrics()

# Requested model/alias names known to the router; used to bound the
# (client-controlled) `alias` label. Anything outside this set buckets to
# "other". Empty set = no bounding (e.g. unit tests).
_KNOWN_ROUTABLE: set[str] = set()


def set_known_routable(names: set[str]) -> None:
    _KNOWN_ROUTABLE.clear()
    _KNOWN_ROUTABLE.update(names)


def metrics_enabled() -> bool:
    return os.environ.get("LLM_RELAY_METRICS", "1").lower() in {"1", "true", "yes", "on"}


def normalize_client(raw: str | None) -> str:
    """Bucket the X-Llm-Relay-Client header to a bounded label set."""
    if not raw:
        return "unknown"
    v = raw.strip().lower()
    if not v or v == "unknown":
        return "unknown"
    if v in _KNOWN_CLIENTS:
        return v
    return "other"


def normalize_alias(raw: str | None) -> str:
    """Bound the (client-controlled) requested model/alias label. Pass through
    known routes; bucket unknown values to "other" once a known set is
    registered (empty set = no bounding)."""
    if not raw:
        return "none"
    if _KNOWN_ROUTABLE and raw not in _KNOWN_ROUTABLE:
        return "other"
    return raw


def did_fall_back(selected_model: str | None, ranked: list[str]) -> bool:
    """True when the served model was not the preferred (first-ranked) candidate
    — i.e. the router fell back. ``ranked`` is ``RouteResult.decision['ranked']``."""
    if not selected_model or not ranked:
        return False
    return selected_model != ranked[0]


def _safe(label: str | None) -> str:
    return label if label else "none"


def _extract_usage(usage: dict | None, response_body: dict | None) -> dict:
    """Token usage lives in the streaming-reassembled ``usage`` dict OR, for
    non-streaming responses, in ``response_body["usage"]``. Handle both."""
    if usage:
        return usage
    if isinstance(response_body, dict):
        u = response_body.get("usage")
        if isinstance(u, dict):
            return u
    return {}


class RelayMetrics:
    """Holds the request/token/fallback/duration collectors against a registry.

    Inject a fresh ``CollectorRegistry`` in tests; production uses the default
    via :func:`get_metrics`.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry if registry is not None else REGISTRY
        self.requests = Counter(
            "llm_relay_requests",
            "Chat-completion requests routed by the relay.",
            ["provider", "model", "alias", "outcome", "client"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "llm_relay_tokens",
            "Prompt/completion tokens routed by the relay.",
            ["provider", "model", "direction", "client"],
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "llm_relay_fallbacks",
            "Requests that fell back off their preferred candidate to another model.",
            ["alias", "model", "client"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "llm_relay_request_duration_seconds",
            "End-to-end relay request duration in seconds.",
            ["provider", "model"],
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )

    def record_request(
        self,
        *,
        alias: str | None,
        model: str | None,
        provider: str | None,
        outcome: str,
        client: str | None,
        usage: dict | None,
        response_body: dict | None,
        duration_s: float | None,
        fell_back: bool,
    ) -> None:
        if not metrics_enabled():
            return
        prov, mdl, ali, cli = _safe(provider), _safe(model), normalize_alias(alias), normalize_client(client)
        self.requests.labels(provider=prov, model=mdl, alias=ali, outcome=outcome, client=cli).inc()

        eff = _extract_usage(usage, response_body)
        pt, ct = eff.get("prompt_tokens"), eff.get("completion_tokens")
        if pt:
            self.tokens.labels(provider=prov, model=mdl, direction="prompt", client=cli).inc(int(pt))
        if ct:
            self.tokens.labels(provider=prov, model=mdl, direction="completion", client=cli).inc(int(ct))

        if duration_s is not None and duration_s >= 0:
            self.duration.labels(provider=prov, model=mdl).observe(duration_s)

        if fell_back:
            self.fallbacks.labels(alias=ali, model=mdl, client=cli).inc()


class DiscoveryCollector:
    """Pull-based collector that reads live backend state from the
    ``DiscoveryManager`` at scrape time. No changes to the poll loop."""

    def __init__(self, discovery: Any):
        self.discovery = discovery

    def collect(self) -> Iterable[GaugeMetricFamily]:
        up = GaugeMetricFamily(
            "llm_relay_backend_up", "1 if backend is healthy/degraded else 0.",
            labels=["backend", "provider"],
        )
        inflight = GaugeMetricFamily(
            "llm_relay_inflight_requests", "In-flight requests per backend.",
            labels=["backend", "provider"],
        )
        cap = GaugeMetricFamily(
            "llm_relay_backend_max_concurrent", "Configured max concurrent slots per backend.",
            labels=["backend", "provider"],
        )
        circuit = GaugeMetricFamily(
            "llm_relay_circuit_breaker_state", "1 if the backend circuit breaker is open else 0.",
            labels=["backend", "provider"],
        )
        clients = getattr(self.discovery, "clients", {}) or {}
        for key, client in clients.items():
            state = getattr(client, "state", None)
            provider = getattr(state, "provider", "") or ""
            status_val = getattr(getattr(state, "status", None), "value", "")
            up.add_metric([key, provider], 1.0 if status_val in ("healthy", "degraded") else 0.0)
            inflight.add_metric([key, provider], float(getattr(client, "inflight_used", 0) or 0))
            mc = getattr(client, "max_concurrent", None)
            cap.add_metric([key, provider], float(mc) if mc else 0.0)
            circuit.add_metric([key, provider], 1.0 if getattr(state, "circuit_open", False) else 0.0)
        yield up
        yield inflight
        yield cap
        yield circuit


_METRICS: RelayMetrics | None = None
_DISCOVERY_COLLECTOR: DiscoveryCollector | None = None


def get_metrics() -> RelayMetrics:
    """Lazily create the singleton bound to RELAY_REGISTRY."""
    global _METRICS
    if _METRICS is None:
        _METRICS = RelayMetrics(RELAY_REGISTRY)
    return _METRICS


def register_discovery_collector(discovery: Any) -> DiscoveryCollector:
    """Register (or replace) the pull-based backend-gauge collector on
    RELAY_REGISTRY. Idempotent across repeated create_app() calls."""
    global _DISCOVERY_COLLECTOR
    if _DISCOVERY_COLLECTOR is not None:
        try:
            RELAY_REGISTRY.unregister(_DISCOVERY_COLLECTOR)
        except Exception:
            pass
    _DISCOVERY_COLLECTOR = DiscoveryCollector(discovery)
    RELAY_REGISTRY.register(_DISCOVERY_COLLECTOR)
    return _DISCOVERY_COLLECTOR


def render_exposition(registry: CollectorRegistry | None = None) -> tuple[bytes, str]:
    """Render the Prometheus exposition as (body, content_type) for a /metrics route.

    A direct route (vs. a mounted ASGI sub-app) avoids the trailing-slash 307
    redirect that would otherwise sit in front of the scrape endpoint."""
    reg = registry if registry is not None else RELAY_REGISTRY
    return generate_latest(reg), CONTENT_TYPE_LATEST
