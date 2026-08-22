"""Prometheus metrics for llm-relay.

Decoupled from the OTLP span path (``instrumentation.py``): these counters and
gauges record request / token / fallback activity plus per-backend health, and
are exposed at ``/metrics`` regardless of whether telemetry (Phoenix) is
enabled or reachable.

Design notes:
- Recording is best-effort and must never raise into the request path. The
  caller wraps ``record_request`` in try/except; label values are coerced to
  safe, bounded strings here (None -> "none"/"unknown"; self-declared client
  names are honored but sanitized and cardinality-capped).
- Backend health gauges are *pull-based*: ``DiscoveryCollector`` reads live
  state off the ``DiscoveryManager`` at scrape time, so the discovery poll loop
  is never touched.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, Counter, Histogram, disable_created_metrics, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from .usage_math import SOURCE_NONE, UsageCounts, resolve_usage

# Always-honored client labels (cap-exempt): never displaced by the cardinality
# cap below, even past the dynamic limit. The repo ships only the generic default
# here; real deployments add their own via LLM_RELAY_KNOWN_CLIENTS (see
# configure_clients_from_env), so no deployment's agent names live in version
# control. This is no longer an allowlist gate — self-identification is honored
# generally (see normalize_client) — it just guarantees these labels a slot.
_DEFAULT_KNOWN_CLIENTS = {"claude-code"}
_KNOWN_CLIENTS = set(_DEFAULT_KNOWN_CLIENTS)

# Cardinality bound for self-declared client labels. Agents identify themselves
# via X-Llm-Relay-Client and the relay records whatever they send (sanitized),
# name-agnostic. To keep the Prometheus `client` series bounded WITHOUT an
# allowlist of names, novel labels are honored only up to this many distinct
# values; beyond it, further new names bucket to "other" so a buggy or hostile
# client cannot explode cardinality. Always-known labels are exempt.
_MAX_DYNAMIC_CLIENTS = 50
_seen_clients: set[str] = set()

# Conservative metric-label charset: a self-declared name is lower-cased and any
# run of other characters collapses to a single '-'.
_CLIENT_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")


def set_known_clients(names: set[str]) -> None:
    """Replace the set of always-honored (cap-exempt) client labels."""
    _KNOWN_CLIENTS.clear()
    _KNOWN_CLIENTS.update(names)


def reset_dynamic_clients() -> None:
    """Clear the runtime set of seen self-declared client labels.

    Production never needs this (the set is bounded by _MAX_DYNAMIC_CLIENTS); it
    exists so tests can assert the cap deterministically without cross-test
    pollution of the module-global cardinality state."""
    _seen_clients.clear()


def _sanitize_client(raw: str) -> str:
    """Lower-case, collapse disallowed chars to '-', trim, and length-bound a
    self-declared client label so the metric series stays well-formed."""
    return _CLIENT_SANITIZE_RE.sub("-", raw.strip().lower()).strip("-")[:32]

# End-to-end latency buckets (seconds): sub-second routing overhead through
# multi-minute large-model generations on the local fleet.
_DURATION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, float("inf"))

# Time-to-first-token buckets (seconds): snappy small models through slow
# prompt-processing on large-context requests, which dominates streaming TTFT.
_TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, float("inf"))

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
    """Resolve the X-Llm-Relay-Client header to a bounded metric label.

    Self-identification is honored as the agent declares it (sanitized), so no
    agent names live in the relay — agents tell the relay who they are. Cardinality
    is bounded generically rather than by an allowlist: always-known labels are
    honored unconditionally, novel labels are honored up to _MAX_DYNAMIC_CLIENTS
    distinct values, and further new names bucket to "other" so a misbehaving
    client cannot explode the series. None / empty / the "unknown" sentinel map to
    "unknown"."""
    if not raw:
        return "unknown"
    v = _sanitize_client(raw)
    if not v or v == "unknown":
        return "unknown"
    if v in _KNOWN_CLIENTS or v in _seen_clients:
        return v
    if len(_seen_clients) < _MAX_DYNAMIC_CLIENTS:
        _seen_clients.add(v)
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
    self-identification, honored as the sanitized value the agent declares, not
    gated by an allowlist); otherwise fall back to a distinctive ``User-Agent``;
    otherwise "unknown"."""
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
          comma-separated labels that are always honored (exempt from the
          self-declared cardinality cap); merged with the built-in default so
          ``claude-code`` stays known. Optional — agents are recorded by their
          self-declared header regardless; this only guarantees a slot.
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
            ["provider", "model", "alias", "outcome", "client", "principal"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "llm_relay_tokens",
            "Input/output tokens routed by the relay. Output INCLUDES reasoning.",
            ["provider", "model", "direction", "client", "principal"],
            registry=self.registry,
        )
        self.reasoning_tokens = Counter(
            "llm_relay_reasoning_tokens",
            "Reasoning (thinking) tokens — an of-which subset of output tokens.",
            ["provider", "model", "client", "principal"],
            registry=self.registry,
        )
        self.usage_source = Counter(
            "llm_relay_usage_source",
            "How each request's token counts were obtained (exact vs estimated).",
            ["source"],
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
            ["provider", "model", "alias", "client"],
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "llm_relay_ttft_seconds",
            "Streaming time-to-first-token (first chunk) in seconds, end-to-end "
            "including routing. Observed only for streamed responses.",
            ["provider", "model", "alias", "client"],
            buckets=_TTFT_BUCKETS,
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "llm_relay_auth_failures",
            "Requests rejected by API-key auth (missing/unknown/disabled key).",
            registry=self.registry,
        )

        self.cache_tokens = Counter(
            "llm_relay_cache_tokens_total",
            "Prompt tokens served from llama.cpp prefix cache (cache_n from timings).",
            ["model", "client", "principal"],
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
        ttft_s: float | None = None,
        principal: str | None = None,
        counts: "UsageCounts | None" = None,
    ) -> None:
        if not metrics_enabled():
            return
        prov, mdl, ali, cli = _safe(provider), _safe(model), normalize_alias(alias), normalize_client(client)
        # Principal cardinality is bounded by the key store (plus
        # internal/anonymous), so no dynamic cap: sanitize only.
        pri = _sanitize_client(principal) if principal else "anonymous"
        self.requests.labels(provider=prov, model=mdl, alias=ali, outcome=outcome, client=cli, principal=pri).inc()

        # Counts come from usage_math so the metrics and the durable store can
        # never disagree. `counts` is passed by the instrumentation layer; the
        # fallback keeps direct callers (and older tests) working.
        if counts is None:
            counts = resolve_usage(usage=usage, response_body=response_body,
                                   streamed=False)
        if counts.input_tokens:
            self.tokens.labels(provider=prov, model=mdl, direction="input",
                               client=cli, principal=pri).inc(counts.input_tokens)
        if counts.output_tokens:
            self.tokens.labels(provider=prov, model=mdl, direction="output",
                               client=cli, principal=pri).inc(counts.output_tokens)
        if counts.reasoning_tokens:
            self.reasoning_tokens.labels(provider=prov, model=mdl,
                                         client=cli, principal=pri).inc(counts.reasoning_tokens)
        if counts.usage_source != SOURCE_NONE:
            self.usage_source.labels(source=counts.usage_source).inc()

        # Prefix-cache reuse: extract cache_n from llama.cpp timings if present.
        cache_n = (response_body or {}).get("timings", {}).get("cache_n", 0)
        if cache_n:
            self.cache_tokens.labels(model=mdl, client=cli, principal=pri).inc(int(cache_n))

        if duration_s is not None and duration_s >= 0:
            self.duration.labels(provider=prov, model=mdl, alias=ali, client=cli).observe(duration_s)

        if ttft_s is not None and ttft_s >= 0:
            self.ttft.labels(provider=prov, model=mdl, alias=ali, client=cli).observe(ttft_s)

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


class PromptStoreCollector:
    """Pull-based gauges for prompt-store growth, read at scrape time.

    These two series are *operational telemetry, not accounting*. Prompt
    retention is indefinite by decision, so growth has to be observable rather
    than discovered by a full disk -- that puts store size and stored-message
    count in the same category as process uptime. Anything used to COMPUTE COST
    (tokens, dollars, per-principal attribution) does NOT belong in a gauge: it
    lives in the usage database and is queried from there. Do not "correct"
    these two into usage_store because usage moved off Prometheus.

    Pull-based like :class:`DiscoveryCollector`: nothing is maintained on the
    capture write path, so a scrape cannot perturb the store. The database is
    opened read-only, and only when it already exists -- a scrape never creates
    a store, and unset ``LLM_RELAY_PROMPT_DB`` (capture off) reports nothing.
    """

    def _families(self) -> tuple[GaugeMetricFamily, GaugeMetricFamily]:
        return (
            GaugeMetricFamily(
                "llm_relay_prompt_store_bytes",
                "On-disk size of the prompt store including WAL sidecars, in bytes.",
            ),
            GaugeMetricFamily(
                "llm_relay_prompt_store_messages",
                "Distinct messages stored (content-addressed: resends do not inflate it).",
            ),
        )

    def describe(self) -> Iterable[GaugeMetricFamily]:
        # Registration asks describe() when present, so registering this
        # collector never touches the filesystem or the database.
        return self._families()

    def collect(self) -> Iterable[GaugeMetricFamily]:
        size, messages = self._families()
        path = os.environ.get("LLM_RELAY_PROMPT_DB", "").strip()
        if not path or not os.path.exists(path):
            # Capture off, or nothing captured yet. The families are still
            # yielded so the series stay documented in the exposition.
            yield size
            yield messages
            return
        # Bytes come from the filesystem rather than a query, so disk growth
        # stays visible even if the read-only open below cannot proceed (a WAL
        # database with no live writer has no -shm sidecar to attach).
        on_disk = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                on_disk += os.path.getsize(path + suffix)
            except OSError:
                pass
        size.add_metric([], float(on_disk))
        try:
            # Imported lazily: the prompt store is optional, and a scrape must
            # not be the reason this module grows an import.
            import sqlite3
            from pathlib import Path

            from . import prompt_store

            conn = sqlite3.connect(
                Path(os.path.abspath(path)).as_uri() + "?mode=ro", uri=True)
            try:
                messages.add_metric(
                    [], float(prompt_store.stats(conn, path)["stored_messages"]))
            finally:
                conn.close()
        except Exception:
            # Best-effort, like the capture path itself: a scrape must never
            # fail because of the prompt store.
            pass
        yield size
        yield messages


_METRICS: RelayMetrics | None = None
_DISCOVERY_COLLECTOR: DiscoveryCollector | None = None
_PROMPT_STORE_COLLECTOR: PromptStoreCollector | None = None


def get_metrics() -> RelayMetrics:
    """Lazily create the singleton bound to RELAY_REGISTRY."""
    global _METRICS
    if _METRICS is None:
        _METRICS = RelayMetrics(RELAY_REGISTRY)
    return _METRICS


def record_auth_failure() -> None:
    """Count one rejected request. Best-effort; never raises into the gate."""
    if not metrics_enabled():
        return
    try:
        get_metrics().auth_failures.inc()
    except Exception:
        pass


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


def register_prompt_store_collector() -> PromptStoreCollector:
    """Register (or replace) the prompt-store growth gauges on RELAY_REGISTRY.

    Registered at import rather than from ``create_app()`` because, unlike
    DiscoveryCollector, this collector binds to no live object: it resolves
    ``LLM_RELAY_PROMPT_DB`` at scrape time, so it costs nothing and reports
    nothing until content capture is switched on. Idempotent, so re-importing
    or re-registering cannot raise a duplicate-series error."""
    global _PROMPT_STORE_COLLECTOR
    if _PROMPT_STORE_COLLECTOR is not None:
        try:
            RELAY_REGISTRY.unregister(_PROMPT_STORE_COLLECTOR)
        except Exception:
            pass
    _PROMPT_STORE_COLLECTOR = PromptStoreCollector()
    RELAY_REGISTRY.register(_PROMPT_STORE_COLLECTOR)
    return _PROMPT_STORE_COLLECTOR


register_prompt_store_collector()


def render_exposition(registry: CollectorRegistry | None = None) -> tuple[bytes, str]:
    """Render the Prometheus exposition as (body, content_type) for a /metrics route.

    A direct route (vs. a mounted ASGI sub-app) avoids the trailing-slash 307
    redirect that would otherwise sit in front of the scrape endpoint."""
    reg = registry if registry is not None else RELAY_REGISTRY
    return generate_latest(reg), CONTENT_TYPE_LATEST
