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
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    preference: float = 0.5
    privacy: Privacy = Privacy.local_only


@dataclass
class ModeConfig:
    description: str = ""
    ports: list[int] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    default: str = ""


@dataclass
class RankingWeights:
    quality: float = 0.4
    latency: float = 0.3
    cost: float = 0.1
    availability: float = 0.2


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
    ranking: RankingWeights = field(default_factory=RankingWeights)
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
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_opened_at: float | None = None


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
