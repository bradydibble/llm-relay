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
TRAFFIC_FRESH_S = 120.0
# A real request that COMPLETED successfully this recently is proof the
# backend is generating. A probe that times out inside that window starved
# behind legitimate traffic (a 512k-context prefill can hold vLLM's scheduler
# past PROBE_TIMEOUT), it did not catch a wedge — a wedged slot completes
# nothing. Without this, heavy long-context load opened the circuit while
# completions were flowing, 503ing every named-model request for minutes at a
# time (2026-08-21 evening, gb200 + llama-01). Kept moderate so a genuine
# wedge right after a burst of traffic still opens within
# TRAFFIC_FRESH_S + 2 probe cycles (~3.5 min) — well inside a wedge's
# multi-hundred-second hang.


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
        """Send a tiny STREAMING completion to one backend with a hard timeout.

        Uses stream=true so the probe gets the first token quickly even on a
        busy model (non-streaming queues behind other requests; streaming gets
        the first token as soon as the model starts generating). The failure
        mode we're catching is a wedged slot — which produces 0 tokens in any
        timeout. A healthy but busy model produces its first token within the
        TTFT deadline even if it's serving other requests.

        Timeout or error = failure. At least 1 token received = success."""
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
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=PROBE_TIMEOUT, write=3.0, pool=3.0)
            ) as c:
                # Use stream to get first token fast, not wait for full response
                async with c.stream("POST", f"{client.base_url}/v1/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    # Read just enough to confirm the model is generating.
                    # We don't need the full response — first SSE chunk with
                    # a content delta proves the slot is not wedged.
                    got_token = False
                    async for chunk in resp.aiter_text():
                        if "data:" in chunk and "[DONE]" not in chunk:
                            # Got a streaming event — model is alive and generating
                            got_token = True
                            break
                    if not got_token:
                        raise RuntimeError("no SSE data received")
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
        # Traffic evidence: a completed real request within TRAFFIC_FRESH_S
        # proves the backend is generating — treat the probe as starved, not
        # the backend as wedged. Also reset the streak so isolated starved
        # probes in separate busy windows never accumulate into an open.
        last_traffic_ns = getattr(client, "last_traffic_success_ns", None)
        if last_traffic_ns and not health.circuit_open:
            age_s = (time.time_ns() - last_traffic_ns) / 1e9
            if age_s < TRAFFIC_FRESH_S:
                health.consecutive_failures = 0
                _log.info(
                    "L2 probe failed for %s but a real request completed %.0fs "
                    "ago — busy backend, not a wedge; failure not counted (%s)",
                    key, age_s, exc,
                )
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
