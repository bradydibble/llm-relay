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
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

# Calling agents we attribute traffic to: bounds the `client` metric label so a
# malformed or novel header value can't explode cardinality (anything else
# buckets to "other"). The repo ships only the generic default below; real
# deployments add their own labels via LLM_RELAY_KNOWN_CLIENTS (see
# configure_clients_from_env), so no deployment's agent names live in version
# control.
_DEFAULT_KNOWN_CLIENTS = {"claude-code"}
_KNOWN_CLIENTS = set(_DEFAULT_KNOWN_CLIENTS)


def set_known_clients(names: set[str]) -> None:
    """Replace the bounded set of known client labels (used by normalize_client)."""
    _KNOWN_CLIENTS.clear()
    _KNOWN_CLIENTS.update(names)

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


# User-Agent substrings that identify a calling agent when no explicit
# X-Llm-Relay-Client header is present. Matched case-insensitively, in order.
# Only agents with a *distinctive* UA belong here; agents whose chat path sends a
# generic SDK User-Agent self-identify via the explicit header instead.
#
# Empty by default in the repo — real deployments add their own
# substring->label patterns via LLM_RELAY_CLIENT_UA_PATTERNS (see
# configure_clients_from_env), so no deployment's agent identifiers live in
# version control.
_UA_CLIENT_PATTERNS: tuple[tuple[str, str], ...] = ()


def set_ua_client_patterns(patterns: tuple[tuple[str, str], ...]) -> None:
    """Replace the User-Agent -> client-label patterns (used by
    client_from_user_agent)."""
    global _UA_CLIENT_PATTERNS
    _UA_CLIENT_PATTERNS = tuple(patterns)


def client_from_user_agent(user_agent: str | None) -> str | None:
    """Map a distinctive User-Agent to a known client label, or None."""
    if not user_agent:
        return None
    ua = user_agent.lower()
    for needle, label in _UA_CLIENT_PATTERNS:
        if needle in ua:
            return label
    return None


def resolve_client(header_value: str | None, user_agent: str | None) -> str:
    """Resolve the calling-agent label for the ``client`` metric dimension.

    An explicit ``X-Llm-Relay-Client`` header wins (intentional
    self-identification, honored even when unrecognized -> "other"); otherwise
    fall back to a distinctive ``User-Agent``; otherwise "unknown"."""
    explicit = normalize_client(header_value)
    if explicit != "unknown":
        return explicit
    return client_from_user_agent(user_agent) or "unknown"


def configure_clients_from_env() -> None:
    """Load deployment-specific client attribution from the environment.

    The repo ships generic defaults (only ``claude-code`` is a known client and
    no User-Agent patterns), so no deployment's agent names live in version
    control. Operators add their own in the (off-repo) service environment:

      ``LLM_RELAY_KNOWN_CLIENTS="claude-code,agent-a,agent-b"``
          comma-separated labels that bound the ``client`` metric dimension;
          merged with the built-in default so ``claude-code`` stays known.
      ``LLM_RELAY_CLIENT_UA_PATTERNS="agent-a-cli=agent-a,agent-b=agent-b"``
          comma-separated ``<ua-substring>=<label>`` pairs; a request with no
          explicit ``X-Llm-Relay-Client`` header whose User-Agent contains the
          substring is attributed to the label.

    Called once at app startup. Idempotent; a missing/empty var leaves the
    corresponding generic default in place.
    """
    known = os.environ.get("LLM_RELAY_KNOWN_CLIENTS", "")
    labels = {c.strip().lower() for c in known.split(",") if c.strip()}
    if labels:
        set_known_clients(_DEFAULT_KNOWN_CLIENTS | labels)

    raw = os.environ.get("LLM_RELAY_CLIENT_UA_PATTERNS", "")
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        needle, _, label = item.partition("=")
        needle, label = needle.strip().lower(), label.strip().lower()
        if needle and label:
            pairs.append((needle, label))
    if pairs:
        set_ua_client_patterns(tuple(pairs))


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

    def collect(self) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
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
        reconciles = CounterMetricFamily(
            "llm_relay_slot_reconciliations",
            "Forced in-flight slot reconciles (leaked-slot containment by the poll loop).",
            labels=["backend", "provider"],
        )
        resets = CounterMetricFamily(
            "llm_relay_backend_resets",
            "Backend resets detected on recovery (circuit recovery or model reload) that wiped in-flight state.",
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
            reconciles.add_metric([key, provider], float(getattr(client, "slot_reconciliations", 0) or 0))
            resets.add_metric([key, provider], float(getattr(client, "backend_resets", 0) or 0))
        yield up
        yield inflight
        yield cap
        yield circuit
        yield reconciles
        yield resets


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
