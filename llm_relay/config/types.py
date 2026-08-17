"""Configuration types for llm-relay."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProviderType(str, Enum):
    """Wire protocol a provider speaks.

    Only ``openai`` exists: the relay forwards OpenAI-compatible chat-completions
    and nothing in the codebase branches on this value. An ``anthropic`` member
    was removed 2026-07-31 along with the Anthropic provider — it was decorative
    (an anthropic-typed provider was forwarded as OpenAI chat-completions anyway),
    and dropping it makes the policy structural: this gateway serves CIQ-operated
    inference, so reintroducing a vendor cloud is a code change and a review, not
    a config edit. `ProviderType("anthropic")` now raises at load.
    """

    openai = "openai"


class Privacy(str, Enum):
    local_only = "local_only"
    cloud_ok = "cloud_ok"


class Confidentiality(str, Enum):
    """Workload sensitivity, declared by the CALLER on each request.

    SCOPE — this axis answers exactly one question: *may this workload run on a
    machine CIQ does not control?* It is a HARDWARE-CUSTODY control, not a
    general data-sensitivity policy. The relay's whole remit is CIQ-operated
    inference (our own open-weight models on our own or borrowed metal), so that
    is the only risk it can speak to: on borrowed hardware the box operator can
    observe or retain anything, and no contract governs them.

    It says NOTHING about sending data to a contracted vendor's inference API.
    That is a different risk model entirely — governed by commercial agreement,
    compliance obligations, and company authorization rather than by who racks
    the machine — and it is deliberately out of this relay's scope. Do not
    generalize this enum into a "can this data leave CIQ" flag.

    ``confidential`` is the DEFAULT and is fail-closed: assume the workload may
    carry CIQ-proprietary material (sales data, Fathom transcripts, Slack,
    closed-source code — Fuzzball, Ledger Pro, ELLM, build/automation tooling),
    so it may only run on hardware CIQ fully owns.

    ``non_confidential`` is an explicit caller assertion that the workload is
    safe to run on metal CIQ does not own — open-source work (kernel, Warewulf,
    Ascender base, public codebases). This is NEVER inferred: an absent or
    unparseable header means ``confidential``. The onus is on the agent operator
    to declare it, and the declaration is clamped against the caller's API-key
    scopes (see ``api.app._clamp_confidentiality``).
    """

    confidential = "confidential"
    non_confidential = "non_confidential"


class Ownership(str, Enum):
    """Who physically controls the machine a provider's models run on.

    Every provider here is CIQ-operated inference — our own models, served by us.
    The only variable is whose rack the GPU sits in.

    ``ciq_owned`` — CIQ controls the machine end to end (llama-01, ciq-l4,
    ciq-mi100). Any workload may run there, confidential or not.

    ``third_party`` — borrowed or shared metal we run our own models on
    (amd-dev, the NVIDIA lab). We control the software; someone else controls
    the machine, and no agreement binds what they may observe or retain. ONLY
    workloads explicitly declared ``non_confidential`` may be routed here.

    NOT what this means: a contracted vendor's inference API (Anthropic, OpenAI).
    Those are a different trust model — the counterparty carries compliance and
    contractual obligations — and the relay does not proxy them at all. Access to
    vendor inference is a company-authorization question handled in the client
    harness, not an ``Ownership`` value.

    Deliberately has NO default: ``providers.yaml`` must state ownership for
    every provider and the loader raises if one omits it. A forgotten tag on
    newly-added borrowed hardware would silently route confidential work onto
    it, so this fails loudly at config load (relay refuses to start, recoverable
    in seconds) rather than quietly at request time (a data leak, which is not).
    """

    ciq_owned = "ciq_owned"
    third_party = "third_party"


class ModelStatus(str, Enum):
    available = "available"
    degraded = "degraded"
    unavailable = "unavailable"
    disabled = "disabled"


class EndpointStatus(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    unavailable = "unavailable"
    disabled = "disabled"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout: int = 30


@dataclass
class ProviderConfig:
    type: ProviderType
    base_url: str
    # Who owns the metal behind this provider. Gates the confidentiality axis:
    # a `confidential` request (the default) is never routed to a `third_party`
    # provider. Required in providers.yaml — see Ownership for why there is no
    # default.
    ownership: Ownership
    enabled: bool = True
    auth_source: str | None = None
    health_endpoint: str = "/v1/models"
    poll_interval: int = 15
    health_check_timeout: int = 5
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    model_overrides: list[str] = field(default_factory=list)
    max_concurrent: int | None = None
    slot_wait_timeout: float = 30.0
    # Extra ports to poll for models that have NO models.yaml entry. Anything a
    # backend on one of these ports reports in /v1/models is auto-discovered and
    # made name-routable (see api.app._reconcile_discovered) — for ad-hoc / bake-off
    # models on unmanaged ports that would otherwise make the host read "down".
    discover_ports: list[int] = field(default_factory=list)


@dataclass
class ModelConfig:
    provider: str
    class_name: str = "unknown"
    port: int | None = None
    path: str = ""
    service: str | None = None  # systemd unit on the provider host; used by llm-mode
    served_model_name: str | None = None  # id the backend reports in /v1/models, when it differs from the config key (e.g. a GGUF filename)
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    preference: float = 0.5
    privacy: Privacy = Privacy.local_only
    # Use-case (category) membership as {use_case: priority}. Model-major config:
    # the loader transposes these into the alias map at load time
    # (aliases[uc] = models tagged uc, sorted by priority desc, then preference,
    # then name). Higher priority = preferred earlier in that category's chain.
    use_cases: dict[str, float] = field(default_factory=dict)
    # Isolation flag. When True the model is reachable ONLY by its exact name
    # (explicit strict request); it is held out of every auto-selection surface
    # (alias open-fallthrough tail and the unknown-id open ranking). Pair with an
    # empty `use_cases` so it is also not a named member of any category. Used for
    # backends an agent must never reach by accident (e.g. a costly/experimental
    # model wired into the relay but gated to manual use).
    manual_only: bool = False
    # Candidate lane. When set, this model only appears as a routing candidate when
    # the request's candidate-lane matches AND is one of 'interactive'/'batch'. An
    # empty or unset value means the model is open to both lanes (default behavior).
    candidate_lane: str | None = None
    # Set on a runtime-discovered model (found on a provider's discover_ports, not
    # in models.yaml). Name-routable only (carries manual_only=True), never
    # persisted, and dropped from the registry when its port stops reporting it.
    discovered: bool = False
    # Variant grouping (plan 2): the logical model this entry is one variant of
    # (e.g. `qwen3-14b` for an AWQ-on-L4 and a Q4-on-MI100 entry), and this
    # variant's precision. Both optional; an entry with no `logical` is a
    # standalone model. Additive: routing still keys on the concrete entry until
    # the dispatcher (plan 3) consumes logical models.
    logical: str | None = None
    quant: str | None = None
    # Request filters (plan 5): keys to strip from, and key/values to set on, the
    # request before it is forwarded to this model's upstream (normalize sampling
    # defaults, drop fields a backend rejects). Empty = no rewrite.
    strip_params: list[str] = field(default_factory=list)
    set_params: dict = field(default_factory=dict)
    # Human-facing one-liner ("what is this model good for") surfaced through
    # /v1/models, /v1/available-models, and the MCP list_models tool so clients
    # and coworkers can pick by description, not just by name.
    description: str | None = None


@dataclass
class CategoryConfig:
    """Per-use-case (category) metadata, keyed by category name under
    ``models.categories``. ``reasoning_floor`` is the opt-in quality gate: a
    minimum ``preference`` a model must clear to serve this category. ``None``
    (the default) means open — any live model in priority order may serve it."""
    reasoning_floor: float | None = None


@dataclass
class ModeConfig:
    description: str = ""
    ports: list[int] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    default: str = ""


@dataclass
class PrivacyConstraints:
    default: Privacy = Privacy.local_only
    cloud_allowed_tags: list[str] = field(default_factory=list)


@dataclass
class FallbackGraph:
    graph: dict[str, list[str]] = field(default_factory=dict)
    retry_on: list[str] = field(default_factory=lambda: ["502", "503", "504", "connection_error"])


@dataclass
class ExplicitBehavior:
    strict: bool = False


@dataclass
class ModeHint:
    when_requesting: str
    unavailable_action: str
    recommended_mode: str | None = None
    alternative: str | None = None
    message: str = ""


@dataclass
class PolicyConfig:
    constraints: PrivacyConstraints = field(default_factory=PrivacyConstraints)
    fallback: FallbackGraph = field(default_factory=FallbackGraph)
    explicit: ExplicitBehavior = field(default_factory=ExplicitBehavior)
    mode_hints: list[ModeHint] = field(default_factory=list)
    # Default max_tokens applied to NON-STREAMING requests when the client set
    # none. Prevents unbounded generation on large-context models (vLLM defaults
    # to max_model_len - prompt = hours). Only applies when the client set NO
    # ceiling; a client-set max_tokens is always forwarded unchanged. Streaming
    # is unaffected (the client sees tokens and can disconnect). Set to 0 to
    # disable (restore unbounded generation); None means not configured (use
    # the code default).
    default_max_tokens: int | None = 8192


@dataclass
class EndpointState:
    provider: str
    status: EndpointStatus = EndpointStatus.healthy
    last_poll: str | None = None
    models: list[str] = field(default_factory=list)
    # Per-model max_model_len reported by the backend (vLLM exposes this on
    # /v1/models). Authoritative metadata source -- keeps the relay accurate
    # when a backend's --max-model-len is changed without a models.yaml edit.
    model_max_lens: dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_opened_at: float | None = None
    # Deliberate maintenance pause (set via the relay's /admin/pause; used by the
    # Reno fleet dashboard scheduler). A paused provider is skipped by the router
    # like a down backend but reported as "paused" (not "down"). paused_until is
    # an ISO8601 string or None (indefinite).
    paused: bool = False
    paused_until: str | None = None
    paused_reason: str | None = None


@dataclass
class ModelState:
    name: str
    provider: str
    status: ModelStatus = ModelStatus.available
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    preference: float = 0.5
    privacy: Privacy = Privacy.local_only


class SaturationError(Exception):
    """Raised when all in-flight slots for a backend are occupied and
    `acquire_slot` exceeds its wait budget.

    Carries a retry_after_seconds hint so the API layer can emit a
    well-formed `Retry-After` HTTP header.
    """

    def __init__(self, backend_key: str, retry_after_seconds: float):
        super().__init__(f"backend {backend_key} saturated; retry after {retry_after_seconds:.1f}s")
        self.backend_key = backend_key
        self.retry_after_seconds = retry_after_seconds


class NoBackendAvailableError(Exception):
    """Raised when no candidate is currently available but the request's
    constraints WOULD be satisfied by a configured model that is merely down or
    paused right now — a transient availability gap, not a genuine mismatch.

    Carries retry_after_seconds so the API can emit a Retry-After header, letting
    batch callers wait and retry through a brief discovery gap or maintenance
    pause instead of treating "No model matches constraints" as terminal.
    Distinct from SaturationError (slots full on a REACHABLE backend) and from a
    genuine no-candidate 503 (no configured model can ever match the constraints).
    """

    def __init__(self, retry_after_seconds: float):
        super().__init__(f"no backend available; retry after {retry_after_seconds:.1f}s")
        self.retry_after_seconds = retry_after_seconds
