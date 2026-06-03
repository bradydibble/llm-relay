"""OpenAI-compatible endpoint client with health checking."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from ..config.types import CircuitBreaker, EndpointState

logger = logging.getLogger(__name__)


def _shared_upstream_bearer() -> str | None:
    """Single Authorization bearer for all upstream probes.

    Homelab convention: every upstream LLM service (vllm `--api-key`,
    llama-server `--api-key`) shares one key. Relay reads that key from the
    env so discovery probes authenticate. Set via the systemd unit using
    LLM_RELAY_UPSTREAM_API_KEY (preferred) or LLM_API_KEY (fallback) — both
    refer to the same ~/.config/llm-relay-api-key file content.
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
    # Monotonic timestamp of the most recent slot acquire. The discovery poll
    # loop uses it to tell a legitimately busy backend from one whose counter is
    # stranded by a leaked slot (inflight_used > 0 but no dispatch in a while).
    last_dispatched_at: float | None = field(default=None, init=False)
    # Cumulative observability counters (surfaced via DiscoveryCollector):
    # forced reconciles of a stuck counter, and detected backend resets that
    # wiped in-flight state.
    slot_reconciliations: int = field(default=0, init=False)
    backend_resets: int = field(default=0, init=False)

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
        # Snapshot pre-poll state so a successful poll can detect a backend that
        # came back from an outage (circuit had tripped) or reloaded (model set
        # changed); either way its pre-outage in-flight slots are dead.
        was_open = self.state.circuit_open
        prev_models = set(self.state.models)
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
                # Fix #3: backend-wipe lifecycle hook. Recovery from a tripped
                # circuit, or a changed model set (a freshly-loaded model),
                # means the backend effectively restarted — wipe stale in-flight
                # accounting that circuit recovery alone wouldn't clear.
                model_set_changed = bool(prev_models) and set(models) != prev_models
                if was_open or model_set_changed:
                    self._on_backend_reset(model_set_changed=model_set_changed)
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

    def reset_inflight(self) -> None:
        """Reset in-flight slot accounting to a clean, fully-idle state.

        Replaces the semaphore so any stranded permits are discarded. A live
        request still holding a SlotHandle on the *old* semaphore releases
        safely: the handle only decrements the counter while it points at the
        active semaphore, so after this swap that release is a counter no-op.
        This is "no corruption," not "no drift" — a reset mid-request can leave
        the counter off by one until the next reconcile cycle, the accepted
        blast-radius tradeoff. Unbounded backends (no semaphore) only zero the
        counter, which they never use.
        """
        self.inflight_used = 0
        if self.max_concurrent is not None and self.max_concurrent > 0:
            self.inflight_sem = asyncio.Semaphore(self.max_concurrent)

    def _on_backend_reset(self, *, model_set_changed: bool) -> None:
        """Wipe in-flight accounting for a backend that just came back from an
        outage or reload. The circuit recovers on its own; this clears the slot
        counters/semaphore that recovery would otherwise leave stranded.

        FAST tier of leaked-slot recovery: fires on the first successful poll
        after a circuit trip (``was_open``) or a model-set change. Sub-threshold
        flaps that never tripped the circuit fall to the SLOW tier,
        ``DiscoveryManager._reconcile_stuck_slots``.
        """
        stuck = self.inflight_used
        self.reset_inflight()
        self.backend_resets += 1
        if stuck > 0:
            reason = "model set changed" if model_set_changed else "recovered from circuit-open outage"
            logger.warning(
                "backend %s (%s) reset (%s): cleared %d stale in-flight slot(s)",
                self.provider_name, self.base_url, reason, stuck,
            )
