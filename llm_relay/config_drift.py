"""Config-drift detection: hashes backend model configs and alerts on unexpected changes.

L4 in the health tier system: L0 (liveness), L1 (decode), L2 (termination),
L3 (canary), L4 (config-drift). Periodically fetches each backend's /v1/models
response, hashes the model IDs + context lengths, and compares against the
last known hash. On drift, logs a warning so operators know a backend's
model config changed (e.g. someone restarted vLLM with different flags, a
model was swapped, context window changed).

This catches the class of bugs where a silent config change causes new
failure modes (e.g. someone removes --tool-call-parser and tool calls
silently stop working, or changes --max-model-len and the context window
shrinks).
"""
from __future__ import annotations

import hashlib
import logging
import asyncio

import httpx

from .config.types import EndpointStatus

_log = logging.getLogger("llm_relay.health")

DRIFT_INTERVAL = 300.0  # check every 5 minutes


class ConfigDriftDetector:
    """Background loop that detects backend config changes via /v1/models hashing."""

    def __init__(self, discovery, config):
        self.discovery = discovery
        self.config = config
        self._hashes: dict[str, str] = {}
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        await asyncio.sleep(DRIFT_INTERVAL)
        while True:
            try:
                await self._check_all()
            except Exception as e:
                _log.error("config-drift loop error: %s", e, exc_info=True)
            await asyncio.sleep(DRIFT_INTERVAL)

    async def _check_all(self):
        tasks = []
        for key, client in self.discovery.clients.items():
            if client.state.status == EndpointStatus.unavailable:
                continue
            if not client.state.models:
                continue
            tasks.append(self._check_one(key, client))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_one(self, key: str, client):
        """Fetch /v1/models from the backend, hash the model config, compare."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
            ) as c:
                resp = await c.get(f"{client.base_url}/v1/models")
                if resp.status_code != 200:
                    return
                data = resp.json()
                # Hash: model IDs + context lengths (the config that matters for routing)
                models = data.get("data", [])
                fingerprint = "|".join(
                    f"{m.get('id','')}:{m.get('max_model_len') or m.get('context_length') or '?'}"
                    for m in sorted(models, key=lambda m: m.get("id", ""))
                )
                h = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

                old = self._hashes.get(key)
                if old is None:
                    self._hashes[key] = h
                    _log.info("config-drift baseline for %s: %s (models=%s)",
                              key, h, [m.get("id") for m in models])
                elif old != h:
                    _log.warning(
                        "CONFIG DRIFT on %s: hash %s -> %s. "
                        "Backend model config changed (model swap, context window, "
                        "or vLLM restart with different flags). "
                        "Old: %s, New: %s",
                        key, old, h,
                        self._hashes[key], h,
                    )
                    self._hashes[key] = h
        except Exception as e:
            _log.debug("config-drift check failed for %s: %s", key, e)
