"""Health polling and model discovery manager."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.types import CircuitBreaker, EndpointState, EndpointStatus, ModelStatus, SaturationError
from .endpoint import TRAFFIC_FRESH_S, EndpointClient, traffic_is_fresh

# Backend-state persistence (observation-first-health-spec §3.5). Cold starts
# used to create real 503 windows on every redeploy: a fresh process had no
# catalog, no status, and no traffic evidence until its first poll cycle —
# under load, exactly when polls starve. Priors newer than PRIORS_MAX_AGE_S
# are adopted at register time as if the restart never happened; anything
# older (or a missing/corrupt file) is ignored and boot behaves as before.
STATE_FILE_VERSION = 1
PRIORS_MAX_AGE_S = 600.0
STATE_FLUSH_INTERVAL_S = 5.0


def _backend_state_path() -> Path | None:
    """Where backend state persists across restarts. Explicit env first; else
    beside the audit log (the unit's one guaranteed-writable state dir); else
    persistence is disabled (tests, bare local runs)."""
    explicit = os.environ.get("LLM_RELAY_STATE_DIR")
    if explicit:
        return Path(explicit) / "backend-state.json"
    audit = os.environ.get("LLM_RELAY_AUDIT_LOG")
    if audit:
        return Path(audit).parent / "backend-state.json"
    return None

logger = logging.getLogger(__name__)


def _default_unavailable_alert_seconds() -> float:
    """Seconds a backend must stay continuously unavailable before the poll
    loop emits a WARNING (default 300s / ~5min). Env-overridable via
    ``LLM_RELAY_UNAVAILABLE_ALERT_SECONDS``. Purely observability: does NOT
    affect availability, routing, or gating.
    """
    raw = os.environ.get("LLM_RELAY_UNAVAILABLE_ALERT_SECONDS")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return 300.0


def _default_reconcile_idle() -> float:
    """Default idle-reconcile window (seconds), env-overridable.

    Reads ``LLM_RELAY_SLOT_RECONCILE_IDLE_SECONDS`` so operators can size the
    window to their longest legitimate job without a code change; a missing,
    malformed, or non-positive value falls back to 1 hour."""
    raw = os.environ.get("LLM_RELAY_SLOT_RECONCILE_IDLE_SECONDS", "")
    try:
        v = float(raw)
        if v > 0:
            return v
    except ValueError:
        pass
    return 3600.0


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
    # Optional override: config model name -> the id the backend reports in
    # /v1/models when they differ (e.g. a GGUF filename). Lets the relay correlate
    # a configured model with what the backend actually serves.
    served_names: dict[str, str] = field(default_factory=dict)
    # Idle window (seconds) after which a bounded backend showing inflight_used
    # > 0 with no recent dispatch is treated as having leaked slots and is
    # force-reconciled. Defaults to 1 hour; override with the
    # LLM_RELAY_SLOT_RECONCILE_IDLE_SECONDS env var (read at construction, so the
    # daemon and CLI both honor it — size the window to your longest job with no
    # code edit).
    #
    # ASSUMPTION: this is a single manager-wide window (not per-client), and it
    # must exceed the longest a *legitimate* single request holds one slot on
    # ANY backend. A job that runs longer (e.g. a batched 200K-context generation
    # on a slow box) would be false-reconciled mid-stream. Thanks to SlotHandle's
    # swap-safe release that is harmless — at worst a transient over-admission by
    # one until the request ends — but it is a spurious reconcile (counter
    # increment + WARNING). Raise the env var for fleets with longer jobs;
    # per-client windows are the escalation if backends' max hold times ever
    # diverge widely.
    slot_reconcile_idle_seconds: float = field(default_factory=_default_reconcile_idle)
    _tasks: list[asyncio.Task] = field(default_factory=list)
    # §3.5 persistence plumbing: priors loaded lazily once, flusher started on
    # first backend registration, last-written body cached so an unchanged
    # snapshot costs a string compare instead of a write.
    _priors: dict | None = field(default=None, init=False)
    _state_flusher_started: bool = field(default=False, init=False)
    _last_state_written: str = field(default="", init=False)

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
        self._apply_prior(key, client)
        if not self._state_flusher_started and _backend_state_path() is not None:
            self._state_flusher_started = True
            self._tasks.append(asyncio.create_task(self._flush_state_loop()))
        self._tasks.append(asyncio.create_task(self._poll_loop(client, poll_interval)))

    def _load_priors(self) -> dict:
        """Read the persisted backend state once per process. Advisory only:
        any problem — missing, corrupt, wrong version, stale — yields {} and
        boot proceeds exactly as it did before persistence existed."""
        if self._priors is not None:
            return self._priors
        self._priors = {}
        path = _backend_state_path()
        if path is None:
            return self._priors
        try:
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict) or doc.get("version") != STATE_FILE_VERSION:
                return self._priors
            saved_at = doc.get("saved_at_ns")
            if not isinstance(saved_at, int):
                return self._priors
            age_s = (time.time_ns() - saved_at) / 1e9
            if age_s < 0 or age_s > PRIORS_MAX_AGE_S:
                return self._priors
            backends = doc.get("backends")
            if isinstance(backends, dict):
                self._priors = backends
        except Exception:
            pass
        return self._priors

    def _apply_prior(self, key: str, client: EndpointClient) -> None:
        entry = self._load_priors().get(key)
        if not isinstance(entry, dict):
            return
        try:
            status = EndpointStatus(entry.get("status", ""))
            models = [m for m in entry.get("models") or [] if isinstance(m, str)]
        except Exception:
            return
        client.state.status = status
        client.state.models = models
        ns = entry.get("last_traffic_success_ns")
        if isinstance(ns, int) and 0 < ns <= time.time_ns():
            client.last_traffic_success_ns = ns
        for m in models:
            self.model_to_client.setdefault(m, key)
        logger.info(
            "adopted persisted priors for backend %s: status=%s, %d model(s) — "
            "restart continuity (spec §3.5)", key, status.value, len(models),
        )

    def _state_snapshot(self) -> dict:
        return {
            "version": STATE_FILE_VERSION,
            "saved_at_ns": time.time_ns(),
            "backends": {
                key: {
                    "status": c.state.status.value,
                    "models": list(c.state.models),
                    "last_traffic_success_ns": c.last_traffic_success_ns,
                }
                for key, c in self.clients.items()
            },
        }

    async def _flush_state_loop(self) -> None:
        """Debounced persistence: every STATE_FLUSH_INTERVAL_S, write the
        snapshot iff its content changed (saved_at excluded from the compare).
        Write-temp-then-rename so a half-written file can never be read as
        state. Failures are logged at debug and never affect serving."""
        path = _backend_state_path()
        if path is None:
            return
        while True:
            await asyncio.sleep(STATE_FLUSH_INTERVAL_S)
            try:
                snap = self._state_snapshot()
                body = json.dumps(snap["backends"], sort_keys=True)
                if body == self._last_state_written:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
                tmp.write_text(json.dumps(snap, sort_keys=True))
                tmp.replace(path)
                self._last_state_written = body
            except Exception as exc:
                logger.debug("backend-state flush failed: %s", exc)

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

    async def _poll_client_once(self, client: EndpointClient) -> None:
        """One L0 poll: fetch /v1/models and update the endpoint's state.

        Extracted from ``_poll_loop`` so the starved-poll rule is testable
        without driving the infinite loop. The rule: an empty or failed poll
        while real completions are flowing (``traffic_is_fresh``) keeps the
        PREVIOUS status and catalog — stale-but-live beats absent. Before
        this, one starved 5s poll flipped a busy backend to unavailable and
        wiped its models, 503ing every named-model request for a machine that
        was serving 200s at that moment. A genuinely dead backend completes
        nothing, so its evidence expires and it flips unavailable as before.
        """
        # Probe demotion (spec §3.2): observation already answers what this
        # poll would ask — the backend completed a real request moments ago
        # and its catalog is known. Skipping the GET entirely also keeps the
        # health plane from competing with real prefills for the thin shared
        # pipe. A catalog change on a BUSY backend waits ≤TRAFFIC_FRESH_S of
        # quiet (accepted in the spec); an empty catalog always polls, so a
        # backend never gets stuck healthy-but-catalogless.
        if traffic_is_fresh(client) and client.state.models:
            client.state.last_poll = datetime.now(timezone.utc).isoformat()
            return
        try:
            models = await client.fetch_models()
            client.state.last_poll = datetime.now(timezone.utc).isoformat()
            if models:
                # Don't override L2-degraded status: the L0 poll proves the
                # process is listening, but L2 may have found a wedged
                # generation slot. Only promote from unavailable -> healthy,
                # not from degraded -> healthy. The L2 probe owns the
                # degraded -> healthy recovery transition.
                if client.state.status != EndpointStatus.degraded:
                    client.state.status = EndpointStatus.healthy
                client.state.models = models
            elif traffic_is_fresh(client) and client.state.models:
                logger.info(
                    "L0 poll starved on %s (%s) but a real request completed "
                    "within %.0fs — keeping the previous catalog",
                    client.provider_name, client.base_url, TRAFFIC_FRESH_S,
                )
            else:
                client.state.status = EndpointStatus.unavailable
                client.state.models = []
        except Exception:
            if not (traffic_is_fresh(client) and client.state.models):
                client.state.status = EndpointStatus.unavailable

    async def _poll_loop(self, client: EndpointClient, interval: int) -> None:
        # Observability only: warn once a backend has been continuously
        # unavailable for >= _default_unavailable_alert_seconds(), and again on
        # recovery. This does NOT touch availability/routing/gating -- a long-
        # down backend is still just ``unavailable`` to the selector; this only
        # makes a silent, prolonged outage visible in the logs.
        alert_after_polls = max(
            1, math.ceil(_default_unavailable_alert_seconds() / max(interval, 1))
        )
        unavailable_streak = 0
        alerted = False
        while True:
            await self._poll_client_once(client)
            # Prolonged-unreachability alert (fires once per outage, plus a
            # recovery line). Tracked in loop-local state so no shared field or
            # routing state is touched.
            if client.state.status == EndpointStatus.unavailable:
                unavailable_streak += 1
                if unavailable_streak == alert_after_polls:
                    alerted = True
                    logger.warning(
                        "backend %s (%s) unreachable for %d consecutive poll(s) "
                        "(>= %.0fs): its models are undiscoverable until it recovers",
                        client.provider_name, client.base_url,
                        unavailable_streak, unavailable_streak * interval,
                    )
            else:
                if alerted:
                    logger.warning(
                        "backend %s (%s) recovered after %d unavailable poll(s) "
                        "(~%.0fs down)",
                        client.provider_name, client.base_url,
                        unavailable_streak, unavailable_streak * interval,
                    )
                unavailable_streak = 0
                alerted = False
            # Containment sweep each cycle: free any slot stranded by a missed
            # release so a leak can't permanently shrink capacity.
            self._reconcile_stuck_slots(client)
            await asyncio.sleep(interval)

    def has_free_slot(self, key: str) -> bool:
        """Whether backend `key` can take a request right now without waiting.

        True when the backend is unbounded (no ``max_concurrent``) or has an
        in-flight slot free. The router uses this to spill past a saturated
        backend WITHOUT paying the per-candidate slot-wait. Unknown keys return
        True (treated as a no-op slot, matching ``acquire_slot``).
        """
        client = self.clients.get(key)
        if client is None:
            return True
        if client.max_concurrent is None or client.max_concurrent <= 0:
            return True
        return client.inflight_used < client.max_concurrent

    def _serves(self, client: EndpointClient, model_name: str) -> bool:
        """Whether *client* currently reports serving *model_name*.

        Matches the backend's reported ids against the model's served name
        (explicit ``served_names`` override, else the config name): exact first,
        then a case-insensitive substring fallback so a llama.cpp backend that
        reports a GGUF filename (e.g. ``Name-UD-Q4_K_XL.gguf``) is still
        recognized. Same convention as the /status ``_actually_available`` check.
        """
        served = self.served_names.get(model_name, model_name)
        reported = client.state.models
        if served in reported or model_name in reported:
            return True
        needles = {served.lower(), model_name.lower()}
        return any(any(n in r.lower() for n in needles) for r in reported)

    def get_model_state(self, model_name: str) -> ModelStatus:
        key = self.model_to_client.get(model_name)
        if key:
            client = self.clients.get(key)
            # Availability is per-MODEL, not per-backend: only trust the mapped
            # client if it is actually serving this model right now (matched via
            # served name / fuzzy). A healthy backend that isn't serving the model
            # (e.g. reimaged with a different served-model-name) must NOT read
            # available — otherwise the router selects it and the upstream 404s.
            # If the mapped client isn't serving it, fall through to any other.
            if client and self._serves(client, model_name):
                if client.state.status == EndpointStatus.healthy:
                    return ModelStatus.available
                if client.state.status == EndpointStatus.degraded:
                    return ModelStatus.degraded
                return ModelStatus.unavailable
            # A configured model is pinned to this exact backend (provider +
            # port + path).  Do not let a similarly-named model on another
            # backend make it appear available: for example, ``model-x`` on
            # provider-a must not fuzzy-match ``model-x-optimized`` on
            # provider-b.  Apart from lying to status consumers, that would
            # select the configured
            # backend even though it is serving a different model.
            return ModelStatus.unavailable
        # Unmapped names are runtime/discovery-only, so searching the reported
        # model ids is the only way to resolve them.
        for client in self.clients.values():
            if self._serves(client, model_name):
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

    # --- Maintenance pause (set via the relay's /admin/pause; used by the Reno
    # fleet dashboard scheduler). A paused provider is skipped by the selector
    # like a down backend, reported as "paused", and the mi100 watchdog stands
    # down -- so a deliberate take-down isn't fought or alarmed. ---
    def pause_provider(self, provider: str, until: str | None = None, reason: str | None = None) -> None:
        for c in self.clients.values():
            if c.provider_name == provider:
                c.state.paused = True
                c.state.paused_until = until
                c.state.paused_reason = reason

    def resume_provider(self, provider: str) -> None:
        for c in self.clients.values():
            if c.provider_name == provider:
                c.state.paused = False
                c.state.paused_until = None
                c.state.paused_reason = None

    def is_provider_paused(self, provider: str) -> bool:
        """True iff `provider` has a live (non-expired) maintenance pause. An
        expired `paused_until` auto-resumes (heals state) and returns False."""
        live = False
        for c in self.clients.values():
            if c.provider_name != provider or not c.state.paused:
                continue
            until = c.state.paused_until
            if until:
                try:
                    dt = datetime.fromisoformat(until)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) >= dt:
                        continue  # expired
                except ValueError:
                    pass  # unparseable -> treat as no expiry (stays paused)
            live = True
        if not live:
            self.resume_provider(provider)  # heal an expired/stale pause
        return live

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
