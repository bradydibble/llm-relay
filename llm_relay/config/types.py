"""Configuration types for llm-relay."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProviderType(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


class Privacy(str, Enum):
    local_only = "local_only"
    cloud_ok = "cloud_ok"


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
    enabled: bool = True
    auth_source: str | None = None
    health_endpoint: str = "/v1/models"
    poll_interval: int = 15
    health_check_timeout: int = 5
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    model_overrides: list[str] = field(default_factory=list)
    max_concurrent: int | None = None
    slot_wait_timeout: float = 30.0


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
