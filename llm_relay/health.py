"""L2 inference health probe: background loop that detects wedged backends.

Sends a tiny completion request to each healthy backend every 30s with a
hard 10s timeout. If the probe times out or returns an error, the backend
is marked degraded and excluded from routing. After 2 consecutive failures,
the backend is marked unavailable (circuit open). A successful probe after
cooldown reopens the circuit (half-open -> healthy).

This catches the failure mode that /v1/models cannot: a backend whose HTTP
listener is alive but whose generation slot is wedged (stuck request,
repetition loop, GPU hang). The L0 health check (/v1/models) passes because
the process is listening; the L2 probe fails because it can't generate.

Architecture (per ChatGPT gpt-5.6 advising):
  - Background task in the relay's lifespan (not per-request)
  - Hard 10-15s timeout on the probe; timeout = unhealthy
  - Immediate circuit-break/exclusion from candidate chain
  - 2 consecutive failures -> unavailable (open circuit)
  - 1 successful probe after cooldown -> healthy (half-open -> closed)
  - 30s interval with jitter to avoid probe stampedes

The relay should NOT restart backends — that's a separate supervisor's job.
The relay's job is to detect the wedge and route around it.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import httpx

from .config.types import EndpointStatus
from .discovery.manager import DiscoveryManager

_log = logging.getLogger("llm_relay.health")

# Probe config
PROBE_INTERVAL = 30.0  # seconds between probe cycles
PROBE_TIMEOUT = 30.0   # hard timeout per probe; timeout = unhealthy.
# 30s is fast enough to catch a wedged slot (hangs for 900s, not 30s) but
# slow enough to avoid false positives on busy reasoning models (a vLLM
# backend with all slots busy can take >15s to admit a probe; a reasoning
# model can spend 20s in thinking before producing 1 token).
PROBE_PROMPT = "OK"
# max_tokens=1: the probe only checks the model can START generating (L1 decode
# smoke). A wedged slot produces 0 tokens in 15s. A healthy reasoning model
# produces at least 1 token in 15s even if full response takes 30s+. Using a
# tiny max_tokens avoids false positives on slow-reasoning models that take
# >10s before their first content token.
PROBE_MAX_TOKENS = 1
FAILURES_TO_OPEN = 2   # consecutive failures before circuit opens
RECOVERY_PROBES = 2    # consecutive successes before circuit closes
COOLDOWN_S = 60.0      # min time between open -> half-open transition


@dataclass
class _BackendHealth:
    """Per-backend circuit breaker state."""
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_ns: int | None = None
    circuit_open: bool = False
    last_probe_ns: int | None = None


class L2HealthProbe:
    """Background L2 termination probe loop. Detects wedged backends that
    pass L0 (/v1/models) but can't generate."""

    def __init__(self, discovery: DiscoveryManager, config):
        self.discovery = discovery
        self.config = config
        self._health: dict[str, _BackendHealth] = {}
        self._task: asyncio.Task | None = None

    def start(self):
        """Start the background probe loop."""
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self):
        """Stop the background probe loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        """Main probe loop: every PROBE_INTERVAL seconds (with jitter),
        probe every backend that L0 reports as healthy."""
        # Wait for the first L0 poll cycle to complete so backends are registered
        await asyncio.sleep(PROBE_INTERVAL + random.uniform(0, 5))
        while True:
            try:
                await self._probe_all()
            except Exception as e:
                _log.error("L2 health probe loop error: %s", e, exc_info=True)
            await asyncio.sleep(PROBE_INTERVAL + random.uniform(0, 5))

    async def _probe_all(self):
        """Probe every backend that is currently healthy (L0 passed).
        Skip backends that are already unavailable (L0 failed)."""
        tasks = []
        for key, client in self.discovery.clients.items():
            if client.state.status == EndpointStatus.unavailable:
                continue  # L0 already says it's down — nothing to add
            # Only probe backends that actually serve models
            if not client.state.models:
                continue
            tasks.append(self._probe_one(key, client))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _probe_one(self, key: str, client):
        """Send a tiny completion to one backend with a hard timeout.
        Timeout or error = failure. Natural termination with short output = success."""
        health = self._health.setdefault(key, _BackendHealth())
        health.last_probe_ns = time.time_ns()

        # Pick the first model this backend serves
        model_name = client.state.models[0] if client.state.models else None
        if not model_name:
            return

        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": PROBE_MAX_TOKENS,
            "temperature": 0,
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=PROBE_TIMEOUT, write=3.0, pool=3.0)
            ) as c:
                resp = await c.post(
                    f"{client.base_url}/v1/chat/completions",
                    json=body,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                data = resp.json()
                ch = data.get("choices", [{}])[0]
                fr = ch.get("finish_reason")
                ct = data.get("usage", {}).get("completion_tokens", 0)
                # Success = model produced at least 1 token. finish_reason
                # can be "length" (capped at 1) or "stop" — both are healthy.
                # The failure mode we're catching is timeout (0 tokens).
                if ct < 1:
                    raise RuntimeError(f"no tokens generated: finish={fr} tokens={ct}")
                # Success
                self._on_success(key, client)
        except Exception as e:
            self._on_failure(key, client, e)

    def _on_success(self, key: str, client):
        health = self._health.get(key)
        if not health:
            return
        # Don't start half-open recovery immediately after the circuit opens —
        # enforce a cooldown so a flaky backend doesn't flap (open/close/open).
        if health.circuit_open and health.last_failure_ns:
            elapsed_s = (time.time_ns() - health.last_failure_ns) / 1e9
            if elapsed_s < COOLDOWN_S:
                return  # still in cooldown; don't count this success yet
        health.consecutive_failures = 0
        health.consecutive_successes += 1

        if health.circuit_open:
            # Half-open: need RECOVERY_PROBES consecutive successes to close
            if health.consecutive_successes >= RECOVERY_PROBES:
                health.circuit_open = False
                if client.state.status == EndpointStatus.degraded:
                    client.state.status = EndpointStatus.healthy
                    _log.info("L2 circuit CLOSED for %s (recovered)", key)
            else:
                _log.info("L2 half-open probe %d/%d succeeded for %s",
                          health.consecutive_successes, RECOVERY_PROBES, key)

    def _on_failure(self, key: str, client, exc: Exception):
        health = self._health.get(key)
        if not health:
            return
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.last_failure_ns = time.time_ns()

        if health.consecutive_failures >= FAILURES_TO_OPEN and not health.circuit_open:
            # Open the circuit: mark degraded so the router skips it
            health.circuit_open = True
            if client.state.status == EndpointStatus.healthy:
                client.state.status = EndpointStatus.degraded
                _log.warning(
                    "L2 circuit OPEN for %s: %d consecutive failures (%s). "
                    "Backend excluded from routing — L0 passes but inference hangs.",
                    key, health.consecutive_failures, exc,
                )
