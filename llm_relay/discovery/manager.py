"""Health polling and model discovery manager."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config.types import CircuitBreaker, EndpointState, EndpointStatus, ModelStatus, SaturationError
from .endpoint import EndpointClient

logger = logging.getLogger(__name__)


class SlotHandle:
    """A single acquired in-flight slot, released synchronously and idempotently.

    Release is intentionally *not* a coroutine: it must be callable from a
    generator ``finally`` or a ``BackgroundTask`` without sitting behind an
    ``await`` that a cancellation storm (client disconnect) could preempt.
    That preemption was the original slot-leak: the release ran last, after
    ``await resp.aclose()``.

    The handle captures the exact semaphore it holds a permit on. If that
    semaphore is later swapped out from under a live request (reconciliation
    or backend-wipe resets), release still frees the permit on the *old*
    semaphore but leaves the live counter alone — so a reset can't make this
    release corrupt the *new* semaphore. This is "no corruption," not "no
    drift": a reset mid-request can leave the counter off by one until the
    next reconcile cycle, which is the accepted blast-radius tradeoff.
    """

    __slots__ = ("_client", "_sem", "_released")

    def __init__(self, client: EndpointClient | None, sem: asyncio.Semaphore | None):
        self._client = client
        self._sem = sem
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._sem is None or self._client is None:
            return
        self._sem.release()
        # Only touch the live counter while this is still the active semaphore.
        # If a reset swapped it, the counter was already zeroed; decrementing
        # would underflow the *new* accounting.
        if self._client.inflight_sem is self._sem:
            self._client.inflight_used = max(0, self._client.inflight_used - 1)


@dataclass
class DiscoveryManager:
    """Track many backends (provider+port/path combos) and per-model availability."""

    clients: dict[str, EndpointClient] = field(default_factory=dict)
    model_to_client: dict[str, str] = field(default_factory=dict)
    # Idle window (seconds) after which a bounded backend showing inflight_used
    # > 0 with no recent dispatch is treated as having leaked slots and is
    # force-reconciled. Default 15 min.
    #
    # ASSUMPTION: this is a single manager-wide window (not per-client), and it
    # must exceed the longest a *legitimate* single request holds one slot on
    # ANY backend. If a backend legitimately holds a slot longer (e.g. a batched
    # 200K-context generation on a slow box > 15 min), that live slot gets
    # false-reconciled mid-stream. Thanks to SlotHandle's swap-safe release this
    # is harmless — at worst a transient over-admission by one until the stream
    # ends — but it is a spurious reconcile (counter increment + WARNING log).
    # Mitigation: raise this default. Per-client windows are the escalation if
    # backends' max hold times ever diverge widely; not worth it today.
    slot_reconcile_idle_seconds: float = 900.0
    _tasks: list[asyncio.Task] = field(default_factory=list)

    async def register_backend(
        self,
        key: str,
        provider_name: str,
        base_url: str,
        models_hint: list[str],
        health_endpoint: str = "/v1/models",
        poll_interval: int = 15,
        circuit_breaker: CircuitBreaker | None = None,
        timeout: float = 5.0,
        max_concurrent: int | None = None,
    ) -> None:
        state = EndpointState(provider=provider_name)
        client = EndpointClient(
            provider_name=provider_name,
            base_url=base_url,
            health_endpoint=health_endpoint,
            timeout=timeout,
            state=state,
            circuit_breaker=circuit_breaker or CircuitBreaker(),
            max_concurrent=max_concurrent,
        )
        self.clients[key] = client
        for m in models_hint:
            self.model_to_client[m] = key
        self._tasks.append(asyncio.create_task(self._poll_loop(client, poll_interval)))

    async def acquire_slot_handle(self, key: str, wait_timeout: float) -> SlotHandle:
        """Acquire an in-flight slot for backend `key`, returning a SlotHandle.

        The caller owns release: call ``handle.release()`` (synchronous,
        idempotent) when the request is done. Used by the streaming path, where
        the slot must outlive the coroutine that acquired it and be releasable
        from a generator ``finally`` / background task without an interruptible
        ``await``.

        If the backend was registered without max_concurrent (or doesn't exist),
        the returned handle is a no-op. Raises SaturationError if no slot becomes
        available within wait_timeout, carrying a retry_after_seconds hint.
        """
        client = self.clients.get(key)
        if client is None or client.inflight_sem is None:
            return SlotHandle(None, None)
        sem = client.inflight_sem
        try:
            await asyncio.wait_for(sem.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError as e:
            raise SaturationError(backend_key=key, retry_after_seconds=wait_timeout) from e
        client.inflight_used += 1
        client.last_dispatched_at = time.monotonic()
        return SlotHandle(client, sem)

    @contextlib.asynccontextmanager
    async def acquire_slot(self, key: str, wait_timeout: float):
        """Acquire an in-flight slot for backend `key`, releasing on exit.

        Thin context-manager wrapper over :meth:`acquire_slot_handle` for the
        non-streaming path, where the slot lifetime matches the ``async with``
        block. If the backend was registered without max_concurrent (or doesn't
        exist), this is a no-op. Raises SaturationError if no slot becomes
        available within wait_timeout, carrying a retry_after_seconds hint.
        """
        handle = await self.acquire_slot_handle(key, wait_timeout)
        try:
            yield
        finally:
            handle.release()

    def _reconcile_stuck_slots(self, client: EndpointClient) -> None:
        """Containment for a leaked in-flight slot.

        If a bounded backend shows ``inflight_used > 0`` but hasn't had a
        dispatch within ``slot_reconcile_idle_seconds``, the counter is almost
        certainly stranded (a slot whose release was missed). Reset the
        accounting so one polling cycle — not forever — is the blast radius.

        This does NOT fix a leak; ``stream_request``'s synchronous release does.
        It only keeps a missed release from permanently shrinking capacity, and
        records the event for observability.

        This is the SLOW tier of leaked-slot recovery (catches anything, within
        one idle window). The FAST tier is ``EndpointClient._on_backend_reset``,
        which wipes immediately on the first successful poll after a circuit trip
        or a model-set change. The only case that depends on this slow tier is a
        leak during a sub-threshold flap — a few poll failures that never tripped
        the circuit, then recovery with the same model set — and with the
        synchronous release in place, even that is unlikely.
        """
        if client.inflight_sem is None or client.max_concurrent is None:
            return
        if client.inflight_used <= 0:
            return
        last = client.last_dispatched_at
        idle = last is None or (time.monotonic() - last) >= self.slot_reconcile_idle_seconds
        if not idle:
            return
        stuck = client.inflight_used
        client.reset_inflight()
        client.slot_reconciliations += 1
        logger.warning(
            "reconciled %d stranded in-flight slot(s) on backend %s (%s): no "
            "dispatch in >= %.0fs; counter + semaphore reset to full capacity",
            stuck, client.provider_name, client.base_url, self.slot_reconcile_idle_seconds,
        )

    async def _poll_loop(self, client: EndpointClient, interval: int) -> None:
        while True:
            try:
                models = await client.fetch_models()
                client.state.last_poll = datetime.now(timezone.utc).isoformat()
                if models:
                    client.state.status = EndpointStatus.healthy
                    client.state.models = models
                else:
                    client.state.status = EndpointStatus.unavailable
                    client.state.models = []
            except Exception:
                client.state.status = EndpointStatus.unavailable
            # Containment sweep each cycle: free any slot stranded by a missed
            # release so a leak can't permanently shrink capacity.
            self._reconcile_stuck_slots(client)
            await asyncio.sleep(interval)

    def get_model_state(self, model_name: str) -> ModelStatus:
        key = self.model_to_client.get(model_name)
        if key:
            client = self.clients.get(key)
            if client:
                if client.state.status == EndpointStatus.healthy:
                    return ModelStatus.available
                if client.state.status == EndpointStatus.degraded:
                    return ModelStatus.degraded
                return ModelStatus.unavailable
        for client in self.clients.values():
            if model_name in client.state.models:
                if client.state.status == EndpointStatus.healthy:
                    return ModelStatus.available
                return ModelStatus.degraded
        return ModelStatus.unavailable

    def get_live_context_window(self, model_name: str) -> int | None:
        """Live max_model_len for `model_name` from the latest /v1/models probe.

        Returns None if no backend currently reports a value (either the model
        isn't being served, the backend is down, or the backend's /v1/models
        response doesn't include `max_model_len`). Callers should fall back to
        the static models.yaml value in that case.
        """
        key = self.model_to_client.get(model_name)
        if key:
            client = self.clients.get(key)
            if client is not None:
                val = client.state.model_max_lens.get(model_name)
                if isinstance(val, int) and val > 0:
                    return val
        for client in self.clients.values():
            val = client.state.model_max_lens.get(model_name)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def get_client_for_model(self, model_name: str) -> EndpointClient | None:
        key = self.model_to_client.get(model_name)
        if key:
            return self.clients.get(key)
        for client in self.clients.values():
            if model_name in client.state.models:
                return client
        return None

    def get_available_models(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for client in self.clients.values():
            for m in client.state.models:
                result[m] = {
                    "provider": client.provider_name,
                    "status": client.state.status.value,
                    "last_poll": client.state.last_poll,
                }
        return result

    def get_endpoint_status(self, key: str) -> EndpointState | None:
        c = self.clients.get(key)
        return c.state if c else None

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
