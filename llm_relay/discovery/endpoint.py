"""OpenAI-compatible endpoint client with health checking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from ..config.types import CircuitBreaker, EndpointState


@dataclass
class EndpointClient:
    """Polls one backend URL (e.g. http://127.0.0.1:8080) for /v1/models."""

    provider_name: str
    base_url: str
    health_endpoint: str = "/v1/models"
    timeout: float = 5.0
    state: EndpointState | None = None
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def __post_init__(self):
        if self.state is None:
            self.state = EndpointState(provider=self.provider_name)

    def _maybe_recover_circuit(self) -> None:
        if not self.state.circuit_open:
            return
        if self.state.circuit_opened_at is None:
            return
        if time.monotonic() - self.state.circuit_opened_at >= self.circuit_breaker.recovery_timeout:
            # Send a probe — leave the breaker closed; if the probe fails we'll
            # immediately trip back open on the next _record_failure.
            self.reset_circuit()

    async def fetch_models(self) -> list[str]:
        self._maybe_recover_circuit()
        if self.state.circuit_open:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}{self.health_endpoint}",
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                models: list[str] = []
                if isinstance(data, dict) and "data" in data:
                    for m in data["data"]:
                        if isinstance(m, dict) and "id" in m:
                            models.append(m["id"])
                elif isinstance(data, list):
                    for m in data:
                        if isinstance(m, str):
                            models.append(m)
                        elif isinstance(m, dict) and "id" in m:
                            models.append(m["id"])
                self.state.consecutive_failures = 0
                self.state.circuit_open = False
                return models
        except Exception:
            self._record_failure()
            return []

    def _record_failure(self) -> None:
        self.state.consecutive_failures += 1
        if self.state.consecutive_failures >= self.circuit_breaker.failure_threshold and not self.state.circuit_open:
            self.state.circuit_open = True
            self.state.circuit_opened_at = time.monotonic()

    def reset_circuit(self) -> None:
        self.state.circuit_open = False
        self.state.circuit_opened_at = None
        self.state.consecutive_failures = 0
