"""OpenAI-compatible endpoint client with health checking."""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx

from ..config.types import CircuitBreaker, EndpointState


def _shared_upstream_bearer() -> str | None:
    """Single Authorization bearer for all upstream probes.

    Homelab convention: every upstream LLM service (vllm `--api-key`,
    llama-server `--api-key`) shares one key. Relay reads that key from the
    env so discovery probes authenticate. Set via the systemd unit using
    LLM_RELAY_UPSTREAM_API_KEY (preferred) or LLM_API_KEY (fallback) — both
    refer to the same /home/admin/.config/llm-relay-api-key file content.
    """
    return os.environ.get("LLM_RELAY_UPSTREAM_API_KEY") or os.environ.get("LLM_API_KEY")


@dataclass
class EndpointClient:
    """Polls one backend URL (e.g. http://127.0.0.1:8080) for /v1/models."""

    provider_name: str
    base_url: str
    health_endpoint: str = "/v1/models"
    timeout: float = 5.0
    state: EndpointState | None = None
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    max_concurrent: int | None = None
    inflight_sem: asyncio.Semaphore | None = field(default=None, init=False)
    # Slots currently held — updated by DiscoveryManager.acquire_slot under the
    # semaphore. Stays at 0 for unbounded backends (no inflight_sem).
    inflight_used: int = field(default=0, init=False)

    def __post_init__(self):
        if self.state is None:
            self.state = EndpointState(provider=self.provider_name)
        if self.max_concurrent is not None and self.max_concurrent > 0:
            self.inflight_sem = asyncio.Semaphore(self.max_concurrent)

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
            headers = {"Accept": "application/json"}
            bearer = _shared_upstream_bearer()
            if bearer:
                headers["Authorization"] = f"Bearer {bearer}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}{self.health_endpoint}",
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                models: list[str] = []
                live_max_lens: dict[str, int] = {}
                if isinstance(data, dict) and "data" in data:
                    for m in data["data"]:
                        if isinstance(m, dict) and "id" in m:
                            mid = m["id"]
                            models.append(mid)
                            # vLLM and most OpenAI-compatible servers publish
                            # max_model_len here; capture for live metadata.
                            mml = m.get("max_model_len")
                            if isinstance(mml, int) and mml > 0:
                                live_max_lens[mid] = mml
                elif isinstance(data, list):
                    for m in data:
                        if isinstance(m, str):
                            models.append(m)
                        elif isinstance(m, dict) and "id" in m:
                            mid = m["id"]
                            models.append(mid)
                            mml = m.get("max_model_len")
                            if isinstance(mml, int) and mml > 0:
                                live_max_lens[mid] = mml
                self.state.model_max_lens = live_max_lens
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
